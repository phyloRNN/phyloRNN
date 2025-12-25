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

# training
EPOCHS = 30
N_ALI_FILES = 10000
W_DIR = "ali_embedding"
LATENT_DIM = 128
BATCH_SIZE = 1
TRAIN = True
predict_training_set = True

# This tells PyTorch to be more efficient with memory segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Manually clear the cache before starting
torch.cuda.empty_cache()



class YInvariantAutoencoder128groupnorm(nn.Module):
    def __init__(self, latent_dim=128):
        super(YInvariantAutoencoder128groupnorm, self).__init__()

        # --- ENCODER ---
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),  # GroupNorm works perfectly with Batch Size 1
            nn.LeakyReLU(0.2),  # Prevents "dead" neurons

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2)
        )

        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))

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
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
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
    return densities




if __name__=="__main__":

    MODEL_PATH = os.path.join(W_DIR, "y_invariant_encoder.pth")

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
            model.train()
            running_train_loss = 0.0

            for batch_n, batch in enumerate(train_loader, 1):
                batch = batch.to(device)
                optimizer.zero_grad()

                reconstruction, _ = model(batch) # returns: reconstruction, embedding
                loss = criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()
                pn.print_update(f"Epoch {epoch + 1} [Train], Batch {batch_n}, Loss: {loss.item():.4f}")

            avg_train_loss = running_train_loss / len(train_loader)

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

            print(f"\nEpoch {epoch + 1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

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

        # fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        # channel_names = ['f(A)', 'f(C)', 'f(T)', 'f(G)', 'f(gap)']
        #
        # for i in range(5):
        #     scatter = axes[i].scatter(
        #         embedding_2d[:, 0],
        #         embedding_2d[:, 1],
        #         c=density_matrix[:, i], # Color by density of current channel
        #         cmap='viridis',
        #         s=10,
        #         alpha=0.6
        #     )
        #     axes[i].set_title(f"Density: {channel_names[i]}")
        #     plt.colorbar(scatter, ax=axes[i])
        #
        # plt.tight_layout()
        # plt.show()


        # 1. Create a dictionary to hold our data
        data_to_save = {
            'file_name': [os.path.basename(f) for f in files],
        }

        # 2. Add the embedding dimensions (e.g., dim_0, dim_1, ...)
        for i in range(matrix.shape[1]):
            data_to_save[f'dim_{i}'] = matrix[:, i]

        # 3. Add our density metadata
        for i in range(5):
            data_to_save[f'density_ch_{i}'] = density_matrix[:, i]

        # 4. Create DataFrame and save
        df = pd.DataFrame(data_to_save)
        df.to_csv(os.path.join(W_DIR, 'embeddings_results.csv'), index=False)
        print("Saved embeddings to embeddings_results.csv")

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
    for i in range(5):
        data_to_save[f'density_ch_{i}'] = density_matrix[:, i]

    # 4. Create DataFrame and save
    df = pd.DataFrame(data_to_save)
    df.to_csv(os.path.join(W_DIR, 'embeddings_results_test.csv'), index=False)
    print("Saved embeddings to embeddings_results_test.csv")

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

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    channel_names = ['freq. A', 'freq C ', 'freq. T', 'freq. G', 'freq. gap']

    for i in range(5):
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






