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

WS = True
TRAIN = False

if WS:
    EPOCHS = 30
    N_ALI_FILES = 10000
    W_DIR = "/local1/dsilvestro/phyloRNN/aliemb"
else:
    EPOCHS = 3
    N_ALI_FILES = 100
    W_DIR = "/Users/dsilvestro/Desktop/ali"

LATENT_DIM = 128
BATCH_SIZE = 8 #64

# This tells PyTorch to be more efficient with memory segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Manually clear the cache before starting
torch.cuda.empty_cache()


class YInvariantAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(YInvariantAutoencoder, self).__init__()

        # ENCODER
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU()
        )
        # Collapse Y-axis (Invariance) and X-axis (Fixed Embedding)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(32, latent_dim)

        # DECODER
        # Project latent back to a spatial feature map
        self.fc_decode = nn.Linear(latent_dim, 32)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 5, 3, padding=1),
            nn.Sigmoid()  # Sigmoid for binary output (0-1 range)
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        return self.fc_encode(x)

    def forward(self, x):
        # 1. Encode to low-dim vector
        latent = self.encode(x)

        # 2. Decode back to 2D
        # For variable sizes, we decode to a fixed "reference" size (e.g., 8x8)
        # or use the input size if available during training.
        x_recon = self.fc_decode(latent).view(-1, 32, 1, 1)

        # Upsample to a standard size or target size
        x_recon = torch.nn.functional.interpolate(x_recon, size=(x.shape[2], x.shape[3]), mode='bilinear')
        x_recon = self.decoder_conv(x_recon)
        return x_recon, latent


class MyBinaryFileDataset(Dataset):
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




# # 1. Initialize your model
# model = YInvariantAutoencoder(latent_dim=LATENT_DIM)
#
# # 2. Print the summary
# # (Batch_size, Channels, Height, Width)
# summary(model, input_size=(100, 5, 32, 64))
#



if __name__=="__main__":

    # Define the path
    MODEL_PATH = os.path.join(W_DIR, "y_invariant_encoder.pth")



    if TRAIN:
        # List of your file paths
        files = glob.glob(os.path.join(W_DIR, "fasta_cds/*"))[:N_ALI_FILES]
        dataset = MyBinaryFileDataset(files)
        # train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=variable_collate_fn)

        # validation split
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_subset, val_subset = random_split(dataset, [train_size, val_size])

        # 2. Create separate loaders
        # Use the same collate_fn you used before if sizes are variable
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)



        # Initialize model
        model = YInvariantAutoencoder(latent_dim=LATENT_DIM)
        summary(model, input_size=(100, 5, LATENT_DIM, BATCH_SIZE))
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Check if CUDA (NVIDIA GPU support) is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if device.type == 'cuda':
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
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
        loaded_model = YInvariantAutoencoder(latent_dim=LATENT_DIM)

        # 2. Load the weights
        loaded_model.load_state_dict(torch.load(MODEL_PATH))

        # 3. Set to evaluation mode
        loaded_model.eval()
        print("Model loaded successfully!")

    # CHECK Y-invariance

    # # check
    # check_invariance(model, files[0])
    #
    #
    # # PREDICT
    # model.eval()
    # with torch.no_grad():
    #     ali_emb = []
    #     for i in range(len(files)):
    #         # 1. Load original data
    #         data_np = parse_file(files[i])  # (5, x, y)
    #         dat = torch.from_numpy(data_np).float().unsqueeze(0)  # Add batch dim
    #
    #         # 3. Get embeddings
    #         ali_emb.append(np.array(model.encode(dat)))
    #         print(f"File: {os.path.basename(files[0])}")
    #
    #
    # pred_loader = DataLoader(dataset, batch_size=100, collate_fn=variable_collate_fn)
    # model.eval()
    # with torch.no_grad():
    #     ali_emb = []
    #     for batch in pred_loader:
    #         # 1. Load original data
    #
    #         # 3. Get embeddings
    #         ali_emb.append(np.array(model.encode(batch)))
    #         print(f"File: {os.path.basename(files[0])}")


    # 1. Extract all embeddings
    all_embeddings = []
    file_labels = [] # Optional: if you have categories for your files

    model.eval()
    with torch.no_grad():
        for f_path in files:
            data = torch.from_numpy(parse_file(f_path)).float().unsqueeze(0)
            latent = model.encode(data.to(device))
            all_embeddings.append(latent.squeeze().numpy())
            # file_labels.append(get_label(f_path))
            pn.print_update(f"File: {f_path}")

    # Convert to a 2D numpy array [num_samples, latent_dim]
    matrix = np.array(all_embeddings)

    # 2. Run UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    matrix_scaled = StandardScaler().fit_transform(matrix)
    embedding_2d = reducer.fit_transform(matrix_scaled)


    # 3. Plot
    # plt.figure(figsize=(10, 7))
    # plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.7, cmap='viridis')
    # plt.title("UMAP Projection of Y-Invariant Embeddings")
    # plt.xlabel("UMAP 1")
    # plt.ylabel("UMAP 2")
    # plt.colorbar()
    # plt.show()


    #

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

    # 3.1 Add UMAP embedding
    data_to_save[f'umap_0'] = embedding_2d[:, 0]
    data_to_save[f'umap_1'] = embedding_2d[:, 1]

    # 4. Create DataFrame and save
    df = pd.DataFrame(data_to_save)
    df.to_csv(os.path.join(W_DIR, 'embeddings_results.csv'), index=False)
    print("Saved embeddings to embeddings_results.csv")

    # TEST SET
    files = glob.glob(os.path.join(W_DIR, "fasta_cds/*"))[N_ALI_FILES:]
    dataset = MyBinaryFileDataset(files)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=variable_collate_fn)

    # 1. Extract all embeddings
    all_embeddings = []
    file_labels = [] # Optional: if you have categories for your files

    model.eval()
    with torch.no_grad():
        for f_path in files:
            data = torch.from_numpy(parse_file(f_path)).float().unsqueeze(0)
            latent = model.encode(data)
            all_embeddings.append(latent.squeeze().numpy())
            # file_labels.append(get_label(f_path))
            pn.print_update(f"File: {f_path}")

    # Convert to a 2D numpy array [num_samples, latent_dim]
    matrix = np.array(all_embeddings)

    # 2. Run UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    matrix_scaled = StandardScaler().fit_transform(matrix)
    embedding_2d = reducer.fit_transform(matrix_scaled)

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

    # 3.1 Add UMAP embedding
    data_to_save[f'umap_0'] = embedding_2d[:, 0]
    data_to_save[f'umap_1'] = embedding_2d[:, 1]

    # 4. Create DataFrame and save
    df = pd.DataFrame(data_to_save)
    df.to_csv(os.path.join(W_DIR, 'embeddings_results_test.csv'), index=False)
    print("Saved embeddings to embeddings_results_test.csv")






