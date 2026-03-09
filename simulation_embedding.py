import torch
import os, glob
import phyloRNN as pn
parse_file = pn.parse_alignment_file_gaps3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sequencoder import *
# training


####------- CHECK AGAINST SIMULATIONS ---------####
W_DIR = "/Users/dsilvestro/Desktop/res128groupnorm/"
bin_dir = "/Users/dsilvestro/Software/phyloRNN-project/phyloRNN/phyloRNN/bin/"
data_dir = os.path.join(W_DIR, "ali/")
res_dir = os.path.join(W_DIR, "simulations/")

# load model
MODEL_PATH = os.path.join(W_DIR, "res25012026/y_invariant_encoder_decorr_attention.pth")
model = YInvariantAutoencoder128groupnorm(latent_dim=LATENT_DIM)
device = torch.device("cpu")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
count_parameters(model)

# SIMULATE DATA with SeqGen
# # sub_models = ['GTRGAMMA', 'JC', 'GTR']
# # [simulate_data_seqgen(data_dir, res_dir, sub_model=m) for m in sub_models]
#
# # SIMULATE DATA with Pyvolve (with empirical ancestral sequence)
# res_dir = os.path.join(W_DIR, "simulations_anc_seq/")
# pn.simulate_data_pyvolve(data_dir, res_dir)
#
# res_dir = os.path.join(W_DIR, "simulations_anc_seq/")
# pn.simulate_data_pyvolve(data_dir, res_dir, sub_model='codon')

res_dir = os.path.join(W_DIR, "simulations_anc_seq/")
# simulate_data_alisim(data_dir, res_dir, sub_model='indel')
pn.simulate_data_alisim(data_dir, res_dir, bin_dir=bin_dir,
                        model_options=['indel', 'anc_seq'], # add:  'CODON' if using codon evol models
                        # evol_model="MG2K{2.0,0.3,0.5}+F1X4+R12", # codon model Omega/Transition/Transversion
                        evol_model="GTR+F+R12", # GTR  + free rates -> add: +I{0.2} for invariant sites
                        # evol_model="ECMK07+F+R12",
                        evol_model_tag="GTR_FR",
                        n_sims=1) #



RES_DIR = os.path.join(W_DIR, "res25012026/sim_res")
os.makedirs(RES_DIR, exist_ok=True)

# sim_files = glob.glob(os.path.join(W_DIR, "simulations/sim_ali_*"))

# load simulated SeqGen alignment files
sub_model = "GTRGAMMA"
sim_files = np.sort(glob.glob(os.path.join(W_DIR, f'simulations/sim_ali_{sub_model}/*.fasta')))
# sim_loader = DataLoader(sim_files, batch_size=BATCH_SIZE, collate_fn=variable_collate_fn)

# load Pyvolve alignment files
sub_model = "pyvolve"
sim_files = np.sort(glob.glob(os.path.join(W_DIR, f'simulations_anc_seq/pyvolve_GTR/*.fasta')))

sub_model = "codon"
sim_files = np.sort(glob.glob(os.path.join(W_DIR, f'simulations_anc_seq/pyvolve_{sub_model}/*.fasta')))

sub_model = "alisim_indel"
sim_files = np.sort(glob.glob(os.path.join(W_DIR, f'simulations_anc_seq/alisim_indel/*.phy')))

sub_model = "alisim_anc_seq"
sim_files = np.sort(glob.glob(os.path.join(W_DIR, f'simulations/{alisim_sub_model}/*.phy')))



# AliGen files
sub_model = ["NT1_NR1_REP1000","NT10_NR100_REP1","NT1000_NR1_REP1"][1]
sim_files = np.sort(glob.glob(os.path.join(W_DIR, f"simulations/{sub_model}", f'NT900*.fasta')))



# 1. Extract all embeddings
all_embeddings_sim = []

model.eval()
with torch.no_grad():
    i = 0
    for f_path in sim_files:
        if "alisim" in sub_model:
            data_np = parse_file(f_path, schema="phylip")
        else:
            data_np = parse_file(f_path)
        data = torch.from_numpy(data_np).float().unsqueeze(0) #.to(device)
        latent = model.encode(data)
        all_embeddings_sim.append(latent.squeeze().cpu().numpy())
        pn.print_update(f"Processed: {os.path.basename(f_path)} ({i + 1} / {len(sim_files)})")
        i += 1

# Convert to a 2D numpy array [num_samples, latent_dim]
data_sim = np.array(all_embeddings_sim)

# Save sim embeddings
data_to_save = {
    'file_name': [os.path.basename(f) for f in sim_files],
}

# 2. Add the embedding dimensions (e.g., dim_0, dim_1, ...)
for i in range(data_sim.shape[1]):
    data_to_save[f'dim_{i}'] = data_sim[:, i]

# 3. Create DataFrame and save
df = pd.DataFrame(data_to_save)
df.to_csv(os.path.join(RES_DIR, f'embeddings_sim_data_{sub_model}.csv'), index=False)
print(f"Saved embeddings to embeddings_sim_data_{sub_model}.csv")

sub_models = ["JC", "GTRGAMMA", "pyvolve", "NT1_NR1_REP1000", "NT10_NR100_REP1", "NT1000_NR1_REP1",
              "codon", "alisim_indel", "alisim_anc_seq"]

sub_model = sub_models[-1]
for sub_model in sub_models:
    print("Processing", sub_model)
    # RUN UMAP
    # 1. Read your saved CSV
    f = os.path.join(W_DIR, "res25012026/embeddings_results.csv")
    data = pd.read_csv(f)
    # 2. Extract the 128 latent variables
    latent_cols = [f'dim_{i}' for i in range(128)]
    X_train = data[latent_cols].values

    # 3. Initialize and FIT the UMAP reducer on the training set
    # We keep the reducer object to transform the test set later
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=123)
    embedding_train = reducer.fit_transform(X_train)

    # Update the dataframe with new UMAP coordinates if desired
    data['umap_0'] = embedding_train[:, 0]
    data['umap_1'] = embedding_train[:, 1]

    # sim data set
    f = os.path.join(RES_DIR, f'embeddings_sim_data_{sub_model}.csv')
    data_sim = pd.read_csv(f)
    latent_cols = [f'dim_{i}' for i in range(128)]
    X_sim = data_sim[latent_cols].values

    # UMAP: using .transform() to project into the same space
    embedding_sim = reducer.transform(X_sim)
    data_sim['umap_0'] = embedding_sim[:, 0]
    data_sim['umap_1'] = embedding_sim[:, 1]


    # Subset the dataframe (to re-plot
    target_files = [os.path.basename(f).split("_sim")[0] + '.fasta' for f in sim_files]
    subset_df = data[data['file_name'].isin(target_files)]

    # fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    # Save the plot
    plt.figure(figsize=(8, 7))
    plt.scatter(data['umap_0'], data['umap_1'], s=12, c="#6baed6", label="All alignments", alpha=0.6)
    plt.scatter(subset_df['umap_0'], subset_df['umap_1'], s=12, color="#08519c", label="Selected alignments", alpha=0.8)
    plt.scatter(data_sim['umap_0'], data_sim['umap_1'], s=12, c="orange", label="Simulated alignments", alpha=0.8)
    plt.legend(loc="best", markerscale=2.0, frameon=True, fontsize='small')
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(f"Empirical UMAP Embedding: {sub_model}")
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, f'umap_sim_projection_{sub_model}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # JOINT UMAP EMBEDDING

    # 1. Load Real Data
    f_real = os.path.join(W_DIR, "res25012026/embeddings_results.csv")
    data = pd.read_csv(f_real)
    latent_cols = [f'dim_{i}' for i in range(128)]
    X_real = data[latent_cols].values

    # 2. Load Simulated Data
    f_sim = os.path.join(RES_DIR, f'embeddings_sim_data_{sub_model}.csv')
    data_sim = pd.read_csv(f_sim)
    X_sim = data_sim[latent_cols].values

    # Stack real and sim data on top of each other
    X_combined = np.vstack([X_real, X_sim])

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=123)
    embedding_combined = reducer.fit_transform(X_combined)

    # 4. Split the coordinates back to their original dataframes
    n_real = len(data)
    data['umap_0'] = embedding_combined[:n_real, 0]
    data['umap_1'] = embedding_combined[:n_real, 1]

    data_sim['umap_0'] = embedding_combined[n_real:, 0]
    data_sim['umap_1'] = embedding_combined[n_real:, 1]

    # Subset real data for specific highlights
    target_files = [os.path.basename(f).split("_sim")[0] + '.fasta' for f in sim_files]
    subset_df = data[data['file_name'].isin(target_files)]

    plt.figure(figsize=(8, 7))
    plt.scatter(data['umap_0'], data['umap_1'], s=10, c="#6baed6", label="All alignments", alpha=0.6)
    plt.scatter(subset_df['umap_0'], subset_df['umap_1'], s=12, color="#08519c", label="Selected alignments", alpha=0.8)
    plt.scatter(data_sim['umap_0'], data_sim['umap_1'], s=10, c="orange", label="Simulated alignments", alpha=0.8)
    plt.legend(loc="best", markerscale=2.0, frameon=True, fontsize='small')
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(f"Joint UMAP Embedding: {sub_model}")
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, f'umap_joint_embedding_{sub_model}.png'), dpi=300, bbox_inches='tight')
    plt.close()



