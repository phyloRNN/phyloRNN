from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Phylo.NewickIO import Writer
import subprocess
import dendropy as dp
import os, glob, re
import numpy as np
constructor = DistanceTreeConstructor()
tree_builder = constructor.nj
from .simulate import simulateDNA
from .utilities import print_update, biopython_msa_from_charmatrix


def simulate_data_seqgen(data_dir, res_dir, bin_dir, sub_model='GTRGAMMA', seed=None, n_sims=1000):

    # test sim ali
    rg = np.random.default_rng(seed)
    all_files = np.sort(glob.glob(os.path.join(data_dir, "fasta_cds/*")))

    files = all_files[rg.choice(range(len(all_files)), size=n_sims, replace=False)]

    os.makedirs(os.path.join(res_dir, 'sim_ali_' + sub_model), exist_ok=True)

    # build NJ tree
    i = 0
    for ali_file in files:
        aln = dp.DnaCharacterMatrix.get(file=open(ali_file), schema='fasta')
        bio_msa = biopython_msa_from_charmatrix(aln)
        calculator = DistanceCalculator('identity')
        dm = calculator.get_distance(bio_msa)
        tree = tree_builder(dm)
        ws = Writer([tree])
        s = [i for i in ws.to_strings()][0]
        t = dp.Tree.get_from_string(s, "newick")
        # t.print_plot()
        # simulate DNA alignment
        sim_aln = simulateDNA(t, seq_length=bio_msa.alignment.shape[1],
                                 subs_model=sub_model,
                                 seqgen_path=os.path.join(bin_dir, "seq-gen"))

        anc_seq = str(bio_msa[0].seq)


        f_name = os.path.basename(ali_file).split(".")[0] + "_sim.fasta"
        sim_aln.write(path=os.path.join(res_dir, 'sim_ali_' + sub_model, f_name), schema='fasta')
        print_update(f"Simulated: {os.path.basename(f_name)} ({i + 1} / {len(files)})")
        # t.write_to_path(os.path.join(W_DIR, 'sim_ali', f_name.replace("_sim.fasta", "_nj.tre")), schema="newick")
        i += 1


def simulate_data_pyvolve(data_dir, res_dir, sub_model='GTR', seed=None, n_sims=1000):
    import pyvolve
    # from Bio import Phlyo
    constructor = DistanceTreeConstructor()
    tree_builder = constructor.nj

    # test sim ali
    rg = np.random.default_rng(seed)
    all_files = np.sort(glob.glob(os.path.join(data_dir, "fasta_cds/*")))

    files = all_files[rg.choice(range(len(all_files)), size=n_sims, replace=False)]

    os.makedirs(os.path.join(res_dir, 'pyvolve_' + sub_model), exist_ok=True)

    # build NJ tree
    i = 0
    ali_file = files[0]
    for ali_file in files:
        aln = dp.DnaCharacterMatrix.get(file=open(ali_file), schema='fasta')
        f_name = os.path.basename(ali_file).split(".")[0] + "_sim.fasta"
        bio_msa = biopython_msa_from_charmatrix(aln)
        calculator = DistanceCalculator('identity')
        dm = calculator.get_distance(bio_msa)
        tree = tree_builder(dm)
        ws = Writer([tree])
        s = [i for i in ws.to_strings()][0]
        t = dp.Tree.get_from_string(s, "newick")
        # t.print_plot()
        print_update(os.path.basename(ali_file))

        tree = pyvolve.read_tree(tree=str(t))
        ali_size = len(str(bio_msa[0].seq))

        if sub_model == 'GTR':
            # 2. Define your substitution model (e.g., Nucleotide GTR or HKY)
            dir_shape_freq = 10.0
            dir_shape_rate = 5.0

            frequencies = list(rg.dirichlet([dir_shape_freq] * 4))
            # 1. Define GTR rates and frequencies
            r = rg.dirichlet([dir_shape_rate] * 6)
            r = list(r / r[-1])
            gtr_rates = {"AC": r[0], "AG": r[1], "AT": r[2], "CG": r[3], "CT": r[4], "GT": r[5]}

            # 2. Add Gamma parameters
            # alpha (shape): Lower values (e.g., 0.5) mean high variation (most sites slow, few very fast)
            # num_categories: Usually 4 or 8 categories are used in phylogenetics
            gtr_gamma_model = pyvolve.Model(
                "nucleotide",
                parameters={
                    "mu": gtr_rates,
                    "state_freqs": frequencies,
                    "alpha": 0.5,  # The 'Gamma' shape parameter
                    "num_categories": 4  # Number of discrete rate bins
                }
            )

            # 3. Define the Ancestral Sequence
            anc_seq = str(bio_msa[0].seq)

            # 4. Create a Partition
            # This links the model to the specific ancestral sequence
            # Note: Pyvolve usually takes length, but we can override the root sequence
            partition = pyvolve.Partition(models=gtr_gamma_model, size=len(anc_seq))

            # 5. The Evolver
            # We pass the partition and tree to the Evolver
            evolver = pyvolve.Evolver(partitions=partition, tree=tree)

            # Inject the ancestral sequence into the root
            # Pyvolve stores partitions in a list; we access the first one.
            evolver.partitions[0].root_sequence = anc_seq
        elif sub_model == 'codon':
            # Define site classes:
            # Class 1: Purifying selection (w=0.1, 60% of sites)
            # Class 2: Neutral evolution (w=1.0, 30% of sites)
            # Class 3: Positive selection (w=3.5, 10% of sites)
            site_params = {
                "omega": [0.1, 1.0, 3.5],
                "proportion": list(rg.dirichlet((10, 5, 0.5)))
            }

            # Apply these classes to the GY94 model
            codon_model = pyvolve.Model("GY94", site_params, heterozygous_sites=True)

               # Define partition and run
            partition = pyvolve.Partition(models=codon_model, size=ali_size)
            evolver = pyvolve.Evolver(partitions=partition, tree=tree)


        elif sub_model == 'indel':

            import re

            # 1. SCALE THE TREE BRANCHES
            # This regex finds all numbers after colons and multiplies them by 100
            def scale_newick(newick, factor=100):
                return re.sub(r':([0-9.]+)', lambda m: f":{float(m.group(1)) * factor:.6f}", newick)

            scaled_newick = scale_newick(str(t))
            tree = pyvolve.read_tree(tree=scaled_newick)

            # 2. Define a substitution model (e.g., GTR or HKY)
            dir_shape_freq = 10.0
            dir_shape_rate = 5.0

            frequencies = list(rg.dirichlet([dir_shape_freq] * 4))
            # 1. Define GTR rates and frequencies
            r = rg.dirichlet([dir_shape_rate] * 6)
            r = list(r / r[-1])
            gtr_rates = {"AC": r[0], "AG": r[1], "AT": r[2], "CG": r[3], "CT": r[4], "GT": r[5]}

            # 2. Add Gamma parameters
            # alpha (shape): Lower values (e.g., 0.5) mean high variation (most sites slow, few very fast)
            # num_categories: Usually 4 or 8 categories are used in phylogenetics
            gtr_gamma_model = pyvolve.Model(
                "nucleotide",
                parameters={
                    "mu": gtr_rates,
                    "state_freqs": frequencies,
                    "alpha": 0.5,  # The 'Gamma' shape parameter
                    "num_categories": 4  # Number of discrete rate bins
                }
            )
            # 3. Define the Indel Model
            # 'rate' is the probability of an indel relative to a substitution
            # 'dist' defines the distribution of indel lengths (usually power-law or geometric)
            indel_params = {
                "rate": 0.5,  # Indel rate relative to substitution rate
                "dist": "geome",  # Geometric distribution of lengths
                "prob": 0.2  # Probability parameter for geometric distribution
            }
            print(indel_params)

            # 4. Set up the Partition
            # 'size' is the starting number of nucleotides before indels occur
            indel_partition = [pyvolve.Partition(
                models=gtr_gamma_model,
                size=ali_size,
                indel_params=indel_params
            )]

            # 5. Run the Simulation
            evolver = pyvolve.Evolver(partitions=indel_partition, tree=tree)

            # 6. Execute the simulation (The object is CALLABLE)
            output_path = os.path.join(res_dir, 'pyvolve_' + sub_model, f_name)

            # f_prefix = os.path.join(res_dir, 'pyvolve_' + sub_model, f_name.replace(".fasta", ""))
            # evolver(seqfile=f_prefix)

            # Do NOT use .apply() or .main(). Just call the object:
            evolver(seqfile=output_path)
        else:
            raise NotImplementedError

def simulate_data_alisim(data_dir, res_dir, bin_dir, model_options=None, seed=None, n_sims=1000,
                         evol_model='GTR+G4+F',
                         # indel settings
                         ins_rate=0.01, del_rate=0.01, mean_len=3, evol_model_tag=None):

    constructor = DistanceTreeConstructor()
    tree_builder = constructor.nj

    # test sim ali
    rg = np.random.default_rng(seed)
    all_files = np.sort(glob.glob(os.path.join(data_dir, "fasta_cds/*")))

    files = all_files[rg.choice(range(len(all_files)), size=n_sims, replace=False)]
    if model_options is None:
        sub_model = ""
    else:
        sub_model = '_'.join(x for x in model_options)
    if evol_model_tag is None:
        evol_model_tag = re.sub(r'[{}+,]', '_', evol_model)


    # Replace any character inside the brackets [] with '_'
    os.makedirs(os.path.join(res_dir, 'alisim_' + sub_model + evol_model_tag), exist_ok=True)

    # build NJ tree
    if seed is None:
        seed = np.random.randint(low=0, high=1000)
    i = 0
    ali_file = files[0]
    for ali_file in files:
        aln = dp.DnaCharacterMatrix.get(file=open(ali_file), schema='fasta')
        f_name = os.path.basename(ali_file).split(".")[0] + "_sim.fa"
        bio_msa = biopython_msa_from_charmatrix(aln)
        calculator = DistanceCalculator('identity')
        dm = calculator.get_distance(bio_msa)
        tree = tree_builder(dm)
        ws = Writer([tree])
        s = [i for i in ws.to_strings()][0]
        t = dp.Tree.get_from_string(s, "newick")
        taxon_names = [taxon.label for taxon in t.taxon_namespace]
        # t.print_plot()
        print_update(os.path.basename(ali_file))
        t_file = os.path.join(res_dir, 'alisim_' + sub_model,'tree.tre')
        t.write(path=t_file, schema='newick')
        anc_sequence_indx = rg.choice(range(len(taxon_names)))

        ali_size = len(str(bio_msa[0].seq))

        cmd = [
            "./iqtree3",
            "--alisim", os.path.join(res_dir, 'alisim_' + sub_model, f_name),
            "-m", evol_model,
            "-t", t_file,
            "--length", str(ali_size),
            "--seed", str(seed + i),
        ]

        if model_options is not None:
            if "indel" in model_options:
                cmd = cmd + ["--indel", "%s,%s" % (ins_rate, del_rate), str(mean_len)]

            if "anc_seq" in model_options:
                cmd = cmd + ["--root-seq", "%s,%s" % (str(ali_file), taxon_names[anc_sequence_indx].replace(" ", "_"))]
            if "CODON" in '_'.join(model_options):
                indx = [i for i in range(len(model_options)) if "CODON" in model_options[i]][0]
                cmd = cmd + ["-st", model_options[indx]]

        cmd = cmd + []

        subprocess.run(cmd, cwd=bin_dir, check=True)
        i += 1

