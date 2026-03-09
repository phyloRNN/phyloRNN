import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os, glob
import phyloRNN as pn
parse_file = pn.parse_alignment_file_gaps3D
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
import umap
from torchinfo import summary
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split
import torch.nn.functional as F
from pathlib import Path
import time, tifffile, sys

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # Tell TF not to hog memory

# training
EPOCHS = 1000
N_ALI_FILES = 10000
try:
    W_DIR = str(Path(__file__).parent / "OrthoMamv12")
except:
    W_DIR = "/Users/dsilvestro/Desktop/res128groupnorm/ali"
    W_DIR = "/Users/dsilvestro/Documents/Projects/Ongoing/GenAli/data/OrthoMamv12/omm_filtered_NT_CDS"

LATENT_DIM = 128
BATCH_SIZE = 1
TRAIN = True
predict_training_set = True

# This tells PyTorch to be more efficient with memory segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Fix fragmentation before starting
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Manually clear the cache before starting
torch.cuda.empty_cache()


class SpatialAttentionPooling(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionPooling, self).__init__()
        # This layer "looks" at the features to decide importance
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        )

    def forward(self, x):
        # x shape: [Batch, Channels, H, W]

        # 1. Compute attention weights
        # attn_weights shape: [Batch, 1, H, W]
        attn_weights = self.attention(x)

        # 2. Normalize weights across space (H and W) using Softmax
        # This ensures all weights sum to 1.0 for each batch
        batch, channels, h, w = x.shape
        attn_weights = attn_weights.view(batch, 1, -1)
        attn_weights = F.softmax(attn_weights, dim=2)
        attn_weights = attn_weights.view(batch, 1, h, w)

        # 3. Weighted Sum: Multiply input by weights and sum over H and W
        # weighted_x shape: [Batch, Channels]
        weighted_x = torch.sum(x * attn_weights, dim=(2, 3))

        return weighted_x


class YInvariantAutoencoder128groupnorm(nn.Module):
    def __init__(self, latent_dim=128):
        super(YInvariantAutoencoder128groupnorm, self).__init__()

        # --- ENCODER ---
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),  # GroupNorm works with Batch Size 1
            nn.LeakyReLU(0.2),  # Prevents "dead" neurons

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2)
        )

        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.attn_pool = SpatialAttentionPooling(in_channels=128)

        self.fc_encode = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, latent_dim)
        )

        # --- DECODER ---
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128)
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 5, 3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = self.attn_pool(x)  # Returns [Batch, 128]
        # x = self.adaptive_pool(x)
        # x = torch.flatten(x, 1)
        return self.fc_encode(x)

    def forward(self, x):
        latent = self.encode(x)
        x_recon = self.fc_decode(latent).view(-1, 128, 1, 1)
        x_recon = torch.nn.functional.interpolate(x_recon, size=(x.shape[2], x.shape[3]), mode='bilinear')
        x_recon = self.decoder_conv(x_recon)
        return x_recon, latent

class SeqBinaryFileDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Call your existing function
        # Expects output shape: (5, x, y)
        data_np = parse_file(file_path)

        # Convert to Torch Tensor
        data_tensor = torch.from_numpy(data_np).float()

        return data_tensor


def decorrelation_loss(z):
    """
    Penalizes correlations between different dimensions in the latent space.
    z: [Batch Size, Latent Dim (128)]
    """
    batch_size, latent_dim = z.shape
    if batch_size <= 1:
        return 0.0  # Cannot decorrelate a single sample

    # 1. Center and Normalize the features (subtract mean)
    z_centered = z - z.mean(dim=0)

    # 2. Calculate the Covariance/Correlation matrix
    # (z_centered.T @ z_centered) results in a [128, 128] matrix
    corr_matrix = (z_centered.T @ z_centered) / (batch_size - 1)

    # 3. Create a mask for off-diagonal elements
    # We want diagonal elements to be 1 (ignored) and off-diagonals to be 0
    diag = torch.eye(latent_dim).to(z.device)

    # Loss = sum of squares of the off-diagonal elements
    off_diag = corr_matrix - (corr_matrix * diag)
    loss = off_diag.pow(2).sum()

    return loss

def variable_collate_fn(batch):
    # This function takes a list of tensors of different sizes
    # and pads them to the max size in the current batch.

    # We need to pad the X and Y dimensions
    # PyTorch's pad_sequence is usually for 1D, so for 2D we do it manually:
    max_x = max([d.shape[1] for d in batch])
    max_y = max([d.shape[2] for d in batch])

    padded_batch = []
    for d in batch:
        pad_x = max_x - d.shape[1]
        pad_y = max_y - d.shape[2]
        # pad format: (left, right, top, bottom) for the last two dims
        padded = torch.nn.functional.pad(d, (0, pad_y, 0, pad_x), value=0)
        padded_batch.append(padded)

    return torch.stack(padded_batch)


def check_invariance(model, file_path):
    model.eval()
    with torch.no_grad():
        # 1. Load original data
        data_np = parse_file(file_path)  # (5, x, y)
        original_tensor = torch.from_numpy(data_np).float().unsqueeze(0)  # Add batch dim

        # 2. Create a vertically shifted version (shifting along the y-axis)
        # We use torch.roll to move rows from top to bottom
        shifted_tensor = torch.roll(original_tensor, shifts=5, dims=3)  # dim 3 is y-axis

        # 3. Get embeddings
        emb_original = model.encode(original_tensor)
        emb_shifted = model.encode(shifted_tensor)

        # 4. Calculate the difference (Cosine Similarity or Euclidean Distance)
        cos_sim = torch.nn.functional.cosine_similarity(emb_original, emb_shifted)
        distance = torch.norm(emb_original - emb_shifted)

        print(f"File: {os.path.basename(file_path)}")
        print(f"Cosine Similarity: {cos_sim.item():.6f} (Should be near 1.0)")
        print(f"Euclidean Distance: {distance.item():.6f} (Should be near 0.0)")


def get_channel_densities(file_path):
    data = parse_file(file_path) # (5, x, y)
    # Calculate mean for each of the 5 channels
    # This represents the percentage of '1's in that channel
    densities = data.mean(axis=(1, 2))
    # add alignment length
    densities = np.append(densities, data.shape[-1])
    return densities


def count_parameters(model):
    # sum up all elements in every parameter tensor
    total_params = sum(p.numel() for p in model.parameters())

    # sum up only those that require gradients (useful if you freeze layers)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    return total_params


if __name__=="__main__":

    MODEL_PATH = os.path.join(W_DIR, "y_invariant_encoder_decorr_attention.pth")

    # Check if CUDA (NVIDIA GPU support) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    if TRAIN:
        # List of your file paths
        files = glob.glob(os.path.join(W_DIR, "fasta_cds/*"))[:N_ALI_FILES]
        dataset = SeqBinaryFileDataset(files)
        # train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=variable_collate_fn)

        # validation split
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_subset, val_subset = random_split(dataset, [train_size, val_size])

        # 2. Create separate loaders
        # Use the same collate_fn you used before if sizes are variable
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=variable_collate_fn)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=variable_collate_fn)

        # Initialize model
        model = YInvariantAutoencoder128groupnorm(latent_dim=LATENT_DIM)
        summary(model, input_size=(100, 5, LATENT_DIM, BATCH_SIZE))
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 2. Move model to GPU
        model.to(device)

        # 3. Initialize optimizer (after moving the model!)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        criterion = torch.nn.BCELoss()

        # Early Stopping Configuration
        patience = 5  # How many epochs to wait for improvement
        counter = 0  # To track how many epochs without improvement
        best_val_loss = float('inf')
        early_stop = False

        for epoch in range(EPOCHS):
            # --- TRAINING PHASE ---
            """
            Training with decorrelation loss (needs batches > 1 to compute correlation)
            But we need batch size of 1 to avoid having to do padding for alignments of different sizes
            """
            model.train()
            running_train_loss = 0.0
            virtual_batch_size = 16
            accumulated_recon_loss = 0.0  # Track sum of recon losses
            lambda_decorr = 0.01

            optimizer.zero_grad()  # Move outside the loop

            # Initialize a running buffer on your GPU
            running_mu = torch.zeros(128).to(device)
            momentum = 0.9  # How much to remember past batches
            diag_mask = torch.eye(128).to(device)
            latent_buffer = []  # To store detached latents for decorrelation

            # ... (Top of loop stays the same) ...

            for batch_n, batch in enumerate(train_loader, 1):
                batch = batch.to(device)

                # 1. Forward Pass
                print(f"DEBUG: Processing file with shape {batch.shape}")
                reconstruction, latent = model(batch)

                # 2. Compute Reconstruction Loss
                recon_loss = criterion(reconstruction, batch)

                # 3. Compute Decorrelation Loss
                d_loss = torch.tensor(0.0).to(device)
                if len(latent_buffer) > 0:
                    past_latents = torch.cat(latent_buffer, dim=0)
                    z_combined = torch.cat([latent, past_latents], dim=0)

                    # Standardize
                    mu = z_combined.mean(dim=0)
                    std = z_combined.std(dim=0) + 1e-8
                    z_std = (z_combined - mu) / std

                    corr_matrix = (z_std.T @ z_std) / (z_std.size(0) - 1)
                    off_diagonals = corr_matrix * (1 - diag_mask)
                    d_loss = off_diagonals.abs().mean()

                # 4. BACKWARD (Memory-Safe)
                total_batch_loss = (recon_loss / virtual_batch_size) + (d_loss * lambda_decorr / virtual_batch_size)
                total_batch_loss.backward()

                # 5. Buffer Management
                with torch.no_grad():
                    running_mu = momentum * running_mu + (1 - momentum) * latent.detach().mean(dim=0)
                    latent_buffer.append(latent.detach())
                    if len(latent_buffer) > virtual_batch_size:
                        latent_buffer.pop(0)

                # 6. Step Optimizer & Activity Monitor (Every 16 batches)
                if batch_n % virtual_batch_size == 0 or batch_n == len(train_loader):
                    # --- NEW: Activity Monitor ---
                    with torch.no_grad():
                        # Check how much each of the 128 dimensions is actually moving
                        # We use the detached buffer we just filled
                        all_latents = torch.cat(latent_buffer, dim=0)
                        std_per_dim = all_latents.std(dim=0)
                        active_dims = (std_per_dim > 1e-4).sum().item()
                        avg_var = std_per_dim.mean().item()
                        max_c = off_diagonals.abs().max().item() if len(latent_buffer) > 1 else 0
                        mean_corr = off_diagonals.abs().mean().item()

                    # Optimizer Step
                    optimizer.step()
                    optimizer.zero_grad()

                    # Logging
                    s = f"Epoch {epoch + 1}, Batch {batch_n}, R-Loss: {recon_loss.item():.3f}"
                    s = s + f" D-Loss: {d_loss.item():.3f} Max/Avg Corr: {max_c:.2f} {mean_corr:.2f}"
                    s = s + f" Active: {active_dims}/128"
                    pn.print_update(s)

                running_train_loss += recon_loss.item()

            # --- VALIDATION PHASE ---
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():  # Disable gradient calculation for speed/memory
                for batch in val_loader:
                    batch = batch.to(device)
                    reconstruction, _ = model(batch)
                    v_loss = criterion(reconstruction, batch)
                    running_val_loss += v_loss.item()

            avg_val_loss = running_val_loss / len(val_loader)
            avg_train_loss = running_train_loss / len(train_loader)

            print(f"\nEpoch {epoch + 1} Summary | Train R-Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # --- EARLY STOPPING LOGIC ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0  # Reset counter
                # Save the best model
                torch.save(model.state_dict(), "best_model.pth")
                print("★ New best model saved!")
            else:
                counter += 1
                print(f"No improvement. EarlyStopping counter: {counter}/{patience}")
                if counter >= patience:
                    print("Early stopping triggered. Training finished.")
                    break

        # Load the best weights back before you start using the model
        model.load_state_dict(torch.load("best_model.pth"))

        # It's best to save just the encoder if that's all you'll use for inference
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model weights saved to {MODEL_PATH}")

    else:
        # 1. Re-initialize the model architecture
        model = YInvariantAutoencoder128groupnorm(latent_dim=LATENT_DIM)

        # 2. Load the weights and move to GPU
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)

        # 3. Set to evaluation mode
        model.eval()
        print("\nModel loaded successfully!")
        files = glob.glob(os.path.join(W_DIR, "fasta_cds/*"))[:N_ALI_FILES]


    # CHECK Y-invariance
    # check_invariance(model, files[0])

    if predict_training_set:
        # 1. Extract all embeddings
        all_embeddings = []
        file_labels = [] # Optional: if you have categories for your files

        with torch.no_grad():
            for f_path in files:
                # Load and prepare data
                data_np = parse_file(f_path)
                data = torch.from_numpy(data_np).float().unsqueeze(0).to(device)

                # Run through model
                latent = model.encode(data)

                # Move result back to CPU for numpy/UMAP
                all_embeddings.append(latent.squeeze().cpu().numpy())

                pn.print_update(f"Processed: {os.path.basename(f_path)}")

        # Convert list to final numpy matrix
        matrix = np.array(all_embeddings)

        # Collect densities for all files
        all_densities = []
        for f_path in files:
            pn.print_update(f"File: {f_path}")
            all_densities.append(get_channel_densities(f_path))

        density_matrix = np.array(all_densities) # Shape: (num_files, 5)

        # 1. Create a dictionary to hold our data
        data_to_save = {
            'file_name': [os.path.basename(f) for f in files],
        }

        # 2. Add the embedding dimensions (e.g., dim_0, dim_1, ...)
        for i in range(matrix.shape[1]):
            data_to_save[f'dim_{i}'] = matrix[:, i]

        # 3. Add our density metadata
        for i in range(6):
            data_to_save[f'density_ch_{i}'] = density_matrix[:, i]

        # 4. Create DataFrame and save
        df = pd.DataFrame(data_to_save)
        df.to_csv(os.path.join(W_DIR, 'embeddings_results.csv'), index=False)
        print("\nSaved embeddings to embeddings_results.csv")

    # TEST SET
    files = glob.glob(os.path.join(W_DIR, "fasta_cds/*"))[N_ALI_FILES:]
    dataset = SeqBinaryFileDataset(files)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=variable_collate_fn)

    # Collect densities for all files
    all_densities = []
    for f_path in files:
        pn.print_update(f"File: {f_path}")
        all_densities.append(get_channel_densities(f_path))

    density_matrix = np.array(all_densities)  # Shape: (num_files, 5)

    # 1. Extract all embeddings
    all_embeddings = []
    file_labels = [] # Optional: if you have categories for your files

    model.eval()
    with torch.no_grad():
        for f_path in files:

            data_np = parse_file(f_path)
            data = torch.from_numpy(data_np).float().unsqueeze(0).to(device)
            latent = model.encode(data)
            all_embeddings.append(latent.squeeze().cpu().numpy())
            pn.print_update(f"Processed: {os.path.basename(f_path)}")

    # Convert to a 2D numpy array [num_samples, latent_dim]
    matrix = np.array(all_embeddings)

    # 1. Create a dictionary to hold our data
    data_to_save = {
        'file_name': [os.path.basename(f) for f in files],
    }

    # 2. Add the embedding dimensions (e.g., dim_0, dim_1, ...)
    for i in range(matrix.shape[1]):
        data_to_save[f'dim_{i}'] = matrix[:, i]

    # 3. Add our density metadata
    for i in range(6):
        data_to_save[f'density_ch_{i}'] = density_matrix[:, i]

    # 4. Create DataFrame and save
    df = pd.DataFrame(data_to_save)
    df.to_csv(os.path.join(W_DIR, 'embeddings_results_test.csv'), index=False)
    print("\nSaved embeddings to embeddings_results_test.csv")

    # RUN UMAP
    # 1. Read your saved CSV
    f = os.path.join(W_DIR, "embeddings_results.csv")
    data = pd.read_csv(f)

    # 2. Extract the 128 latent variables
    # Assuming columns are named dim_0, dim_1, ..., dim_127
    latent_cols = [f'dim_{i}' for i in range(128)]
    X_train = data[latent_cols].values

    # 3. Initialize and FIT the UMAP reducer on the training set
    # We keep the reducer object to transform the test set later
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=123)
    embedding_train = reducer.fit_transform(X_train)

    # Update the dataframe with new UMAP coordinates if desired
    data['umap_0'] = embedding_train[:, 0]
    data['umap_1'] = embedding_train[:, 1]

    # test set
    f = os.path.join(W_DIR, 'embeddings_results_test.csv')
    data_test = pd.read_csv(f)
    latent_cols = [f'dim_{i}' for i in range(128)]
    X_test = data_test[latent_cols].values

    # UMAP: using .transform() to project into the same space
    embedding_test = reducer.transform(X_test)
    data_test['umap_0'] = embedding_test[:, 0]
    data_test['umap_1'] = embedding_test[:, 1]

    fig, axes = plt.subplots(2, 6, figsize=(30, 10))
    channel_names = ['freq. A', 'freq C ', 'freq. T', 'freq. G', 'freq. gap', 'ali. length']

    for i in range(6):
        scatter = axes[0][i].scatter(
            data['umap_0'],
            data['umap_1'],
            c=data[f'density_ch_{i}'],  # Color by density of current channel
            cmap='viridis',
            s=10,
            alpha=0.6
        )
        axes[0][i].set_title(f"{channel_names[i]}")
        plt.colorbar(scatter, ax=axes[0][i])
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")

    for i in range(5):
        scatter = axes[1][i].scatter(
            data_test['umap_0'],
            data_test['umap_1'],
            c=data_test[f'density_ch_{i}'],  # Color by density of current channel
            cmap='viridis',
            s=10,
            alpha=0.6
        )
        axes[1][i].set_title(f"Test set: {channel_names[i]}")
        plt.colorbar(scatter, ax=axes[1][i])
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(W_DIR, 'umap_test_projection.png'), dpi=300, bbox_inches='tight')

    # plt.show()






    ## PLOT MODEL ARCHITECTURE
    # dummy_input = torch.randn(batch_size, channels, num_taxa, seq_length).to(device)
    dummy_input = torch.randn(1, 5, 25, 100).to(device)
    torch.onnx.export(model, dummy_input, os.path.join(W_DIR, "model.onnx"))

    from torchview import draw_graph
    # 1. Initialize your specific model
    # Ensure your class definition for YInvariantAutoencoder128groupnorm
    # and SpatialAttentionPooling are in the script or imported.
    # We'll use a standard "Batch of 1" with 5 DNA channels,
    # 20 taxa (rows), and 500 sites (columns).
    dummy_input = torch.randn(1, 5, 20, 500).to(device)

    # 3. Generate the Graph
    model_graph = draw_graph(
        model,
        input_data=dummy_input,
        graph_name="Sequencoder_ArchitectureLR",
        graph_dir='LR',          # 'TB' for Top-Bottom, 'LR' for Left-Right
        expand_nested=True,
        depth=4,
        show_shapes=True,
        device=device
    )

    # 4. Save to PDF
    # This will create 'Sequencoder_Architecture.pdf' in your current directory
    model_graph.visual_graph.render(format="pdf", directory=W_DIR,cleanup=True)

    count_parameters(model)


    ### check attention map
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    import numpy as np

    W_DIR = "/Users/dsilvestro/Desktop/res128groupnorm/"
    RES_DIR = os.path.join(W_DIR, "res25012026/sim_res")
    MODEL_PATH = os.path.join(W_DIR, "res25012026/y_invariant_encoder_decorr_attention.pth")

    # Check if CUDA (NVIDIA GPU support) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")


    # List of your file paths
    f = np.sort(glob.glob(os.path.join(W_DIR, "ali/fasta_cds/*")))[0]
    files  = [f, f, f]
    dataset = SeqBinaryFileDataset(files)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=variable_collate_fn)

    # Initialize model
    # 1. Re-initialize the model architecture
    model = YInvariantAutoencoder128groupnorm(latent_dim=LATENT_DIM)
    # 2. Load the weights and move to GPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    # 3. Set to evaluation mode
    model.eval()

    def hook_fn(module, input, output):
        # output is the result of self.attention(x) before the Softmax
        captured_weights.append(output.detach())


    # 1. Setup
    num_samples = 3
    captured_weights = []
    data_iter = iter(data_loader)
    model.eval()

    handle = model.attn_pool.attention.register_forward_hook(hook_fn)

    # 2. Collect samples (since batch size is 1, we loop the loader)
    plot_data = []
    for _ in range(num_samples):
        try:
            batch = next(data_iter).to(device)
            with torch.no_grad():
                _ = model(batch)

            # Pull the last weight added to the list
            raw_attn = captured_weights[-1]
            b, c, h, w = raw_attn.shape

            # Softmax over the spatial dimensions (H*W)
            attn_norm = F.softmax(raw_attn.view(b, 1, -1), dim=2).view(h, w).cpu().numpy()
            plot_data.append(attn_norm)
        except StopIteration:
            break

    # 3. Plotting
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3 * num_samples))

    # Handle the case where axes isn't a list (if num_samples = 1)
    if num_samples == 1: axes = [axes]

    for i, attn_map in enumerate(plot_data):
        im = axes[i].imshow(attn_map, aspect='auto', cmap='magma')  # 'magma' is often clearer for small peaks
        axes[i].set_title(f"file: {os.path.basename(files[i])} | Attention Map")
        axes[i].set_ylabel("Taxa")
        fig.colorbar(im, ax=axes[i], label="Weight")

    plt.tight_layout()
    plt.xlabel("Sequence Position (Sites)")
    plt.show()

    # 4. Cleanup
    handle.remove()
