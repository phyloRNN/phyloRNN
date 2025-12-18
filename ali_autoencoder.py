import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os, glob
import phyloRNN as pn
parse_file = pn.parse_alignment_file_gaps3D
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


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



# List of your file paths
files = glob.glob("/Users/dsilvestro/Desktop/fasta_cds/*")[:200]
dataset = MyBinaryFileDataset(files)
train_loader = DataLoader(dataset, batch_size=32, collate_fn=variable_collate_fn)

# Initialize model
model = YInvariantAutoencoder(latent_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCELoss()

for epoch in range(3):
    batch_n = 1
    for batch in train_loader:
        # Move batch to GPU if available
        # batch = batch.to('cuda')

        optimizer.zero_grad()

        # Forward pass
        reconstruction, embedding = model(batch)

        # Loss: Compare reconstructed binary grid to original input
        loss = criterion(reconstruction, batch)

        loss.backward()
        optimizer.step()
        pn.print_update(f"Epoch {epoch + 1}, batch {batch_n}, Avg Loss: {loss:.4f}")
        batch_n += 1
    print(f"\nEpoch {epoch + 1}, Avg Loss: {loss:.4f}")




# CHECK Y-invariance

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

# check
check_invariance(model, files[0])


# PREDICT
model.eval()
with torch.no_grad():
    ali_emb = []
    for i in range(len(files)):
        # 1. Load original data
        data_np = parse_file(files[i])  # (5, x, y)
        dat = torch.from_numpy(data_np).float().unsqueeze(0)  # Add batch dim

        # 3. Get embeddings
        ali_emb.append(np.array(model.encode(dat)))
        print(f"File: {os.path.basename(files[0])}")


pred_loader = DataLoader(dataset, batch_size=100, collate_fn=variable_collate_fn)
model.eval()
with torch.no_grad():
    ali_emb = []
    for batch in pred_loader:
        # 1. Load original data

        # 3. Get embeddings
        ali_emb.append(np.array(model.encode(batch)))
        print(f"File: {os.path.basename(files[0])}")


# 1. Extract all embeddings
all_embeddings = []
file_labels = [] # Optional: if you have categories for your files

model.eval()
with torch.no_grad():
    for f_path in files[:200]:
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


# 3. Plot
plt.figure(figsize=(10, 7))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.7, cmap='viridis')
plt.title("UMAP Projection of Y-Invariant Embeddings")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.colorbar()
plt.show()


#
def get_channel_densities(file_path):
    data = parse_file(file_path) # (5, x, y)
    # Calculate mean for each of the 5 channels
    # This represents the percentage of '1's in that channel
    densities = data.mean(axis=(1, 2))
    return densities

# Collect densities for all files
all_densities = []
for f_path in files:
    pn.print_update(f"File: {f_path}")
    all_densities.append(get_channel_densities(f_path))

density_matrix = np.array(all_densities) # Shape: (num_files, 5)

fig, axes = plt.subplots(1, 5, figsize=(25, 5))
channel_names = ['f(A)', 'f(C)', 'f(T)', 'f(G)', 'f(gap)']

for i in range(5):
    scatter = axes[i].scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=density_matrix[:, i], # Color by density of current channel
        cmap='viridis',
        s=10,
        alpha=0.6
    )
    axes[i].set_title(f"Density: {channel_names[i]}")
    plt.colorbar(scatter, ax=axes[i])

plt.tight_layout()
plt.show()
