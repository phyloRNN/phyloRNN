from .plots import *
from .utilities import *
from .parse_data import *
from .rnn_builder import *
from .sequence_simulator import *
import numpy as np
import random
import sqlite3
import zlib
import multiprocessing
from datetime import datetime
import json


# Function to convert numpy data types to native Python data types
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def simulate_parallel(sim_obj,
                      add_day_tag=True,
                      return_outname=False):
    if add_day_tag:
        day_tag = datetime.now().strftime('%Y%m%d')
    else:
        day_tag = ""
    data = sim_obj.data_name

    print("\nRunning", data)
    list_seeds = np.arange(sim_obj.base_seed, sim_obj.base_seed + sim_obj.CPUs)
    try:
        os.mkdir(sim_obj.ali_path)
    except(OSError):
        pass


    if sim_obj.run_phyml:
        ali_file = os.path.join(sim_obj.ali_path, "ali")
        list_args = [[l, sim_obj.n_sims, ali_file + str(l) + ".phy", True] for l in list_seeds]
    else:
        list_args = [[l, sim_obj.n_sims, False, False] for l in list_seeds]

    if sim_obj.DEBUG:
        print(list_seeds)
        print(len(list_seeds))
        print(list_args)

    if sim_obj.CPUs > 1:
        pool = multiprocessing.Pool()
        res = pool.map(sim_obj.run_sim, list_args)
        pool.close()
    else:
        res = [sim_obj.run_sim(list_args[0])]
    features_ali = []
    features_tree = []
    labels_rates = []
    labels_smodel = []
    labels_tl = []
    info = []

    for i in range(sim_obj.CPUs):
        features_ali = features_ali + res[i][0]
        features_tree = features_tree + res[i][1]
        labels_rates = labels_rates + res[i][2]
        labels_smodel = labels_smodel + res[i][3]
        labels_tl = labels_tl + res[i][4]
        info = info + res[i][5]

    # save arrays
    if sim_obj.min_rate:
        min_rate_tag = "_r" + str(np.log10(sim_obj.min_rate))
    else:
        min_rate_tag = ""

    if sim_obj.format_output in  ('sqlite', 'both'):

        database = "{}{}.db".format(data, day_tag)
        create_table = """
                CREATE TABLE IF NOT EXISTS simulation (
                    sim_id INTEGER PRIMARY KEY,
                    features_ali BLOB,
                    features_tree BLOB,
                    labels_rates BLOB,
                    labels_smodel BLOB,
                    labels_tl BLOB,
                    info TEXT
                )
            """

        insert_simulation = """
                    INSERT INTO simulation (features_ali, features_tree, labels_rates, labels_smodel, labels_tl, info)
                    VALUES (?, ?, ?, ?, ?, ?)
                """

        try:
            with sqlite3.connect(database) as conn:

                cursor = conn.cursor()
                cursor.execute(create_table)

                # Query to get the list of all tables
                #cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                #cursor.fetchall()

                features_ali = np.ascontiguousarray(features_ali)
                features_tree = np.ascontiguousarray(features_tree)
                labels_rates = np.ascontiguousarray(labels_rates)
                labels_smodel = np.ascontiguousarray(labels_smodel)
                labels_tl = np.ascontiguousarray(labels_tl)


                for i in range(len(features_ali)):

                    cursor.execute(insert_simulation, (
                        zlib.compress(np.array(features_ali[i]).tobytes()),
                        zlib.compress(np.array(features_tree[i]).tobytes()),
                        zlib.compress(np.array(labels_rates[i]).tobytes()),
                        zlib.compress(np.array(labels_smodel[i]).tobytes()),
                        zlib.compress(np.array(labels_tl[i]).tobytes()),
                        json.dumps(info[i], default=convert_numpy_types),
                    ))

                # Query to get all rows from the simulation_data table
                #cursor.execute("SELECT * FROM simulation")
                #cursor.fetchall()

                conn.commit()

        except sqlite3.OperationalError as e:
            print(e)

    if sim_obj.format_output in ('npz', 'both'):

        np.savez_compressed(
            file="%s%s.npz" % (data, day_tag),
            features_ali=np.array(features_ali),
            features_tree=np.array(features_tree),
            labels_rates=np.array(labels_rates),
            labels_smodel=np.array(labels_smodel),
            labels_tl=np.array(labels_tl),
            info=np.array(info))

    if return_outname:
        return "%s%s.npz" % (data, day_tag)



class simulator():
    def __init__(self,
                 CPUs = 50,
                 n_sims = 200,  # n. simulations per worker
                 data_name = "training_data", # "compare_data" "test_data"
                 n_taxa = 50,
                 n_sites = 1000,
                 n_eigen_features = 3,
                 min_rate = 0,  #
                 freq_uncorrelated_sites = 0.5,
                 freq_mixed_models = 0.5,
                 p_heterogeneity_model = None, # # ["Gamma", "Bimodal", "GBM", "Spike-and-slab", "Codon"]
                 store_mixed_model_info = False,
                 tree_builder = 'nj',  # 'upgma'
                 subs_model_per_block = False,  # if false same subs model for all blocks
                 phyml_path = None,  # os.path.join(os.getcwd(), "phylogeNN")
                 seqgen_bin = 'seq-gen',
                 seqgen_path = None,  # os.path.join(os.getcwd(), "phylogeNN", )
                 ali_path = None,
                 run_phyml=False,
                 DEBUG=False,
                 verbose = False,
                 base_seed = None,
                 min_avg_br_length=0.0002,
                 max_avg_br_length=0.2,
                 ali_schema="phylip",
                 format_output='sqlite', # 'sqlite' , 'npz, or 'both'
                 ):
        self.DEBUG = DEBUG
        self.format_output=format_output
        self.verbose = verbose
        self.base_seed = base_seed
        self.base_seed = base_seed
        self.CPUs = CPUs
        self.n_sims = n_sims
        self.data_name = data_name
        self.n_taxa = n_taxa
        self.n_sites = n_sites
        self.n_eigen_features = n_eigen_features
        self.min_rate = min_rate
        self.freq_uncorrelated_sites = freq_uncorrelated_sites
        self.freq_mixed_models = freq_mixed_models
        self.p_heterogeneity_model = p_heterogeneity_model
        self.store_mixed_model_info = store_mixed_model_info
        self.tree_builder = tree_builder
        self.subs_model_per_block = subs_model_per_block
        self.phyml_path = phyml_path
        if seqgen_path is not None:
            self.seqgen_path = os.path.join(seqgen_path, seqgen_bin)
        else:
            self.seqgen_path = None
        if ali_path is None:
            ali_path = os.path.join(os.getcwd(), "phyloRNN", "ali_tmp")
        self.ali_path = ali_path
        self.run_phyml = run_phyml
        self.min_avg_br_length = min_avg_br_length
        self.max_avg_br_length = max_avg_br_length
        self.ali_schema = ali_schema

    def reset_prms(self, CPUs, n_sims, data_name, base_seed, run_phyml=None):
        self.CPUs = CPUs
        self.n_sims = n_sims
        self.data_name = data_name
        self.base_seed = base_seed
        if run_phyml is not None:
            self.run_phyml = run_phyml

    def run_sim(self, args):
        [init_seed, n_sims, save_ali, run_phyml_estimation] = args
        seed = random.randint(0, 1000) + init_seed
        rs = get_rnd_gen(seed)
        features_ali = []
        features_tree = []
        labels_rates = []
        labels_smodel = []
        labels_tl = []
        info = []
        subs_models = np.array(['JC', 'HKY', 'GTR'])
        for sim_i in range(n_sims):
            if rs.random() < self.freq_uncorrelated_sites:
                rate_m = "uncorrelated"
                blocks = self.n_sites
                sites_indices = np.arange(blocks)
            else:
                if np.random.random() < self.freq_mixed_models:
                    rate_m = "mixed_model"
                    blocks = 2 # np.min([np.random.geometric(p=0.1), n_sites])  # mean = 10
                    sites_indices = np.sort(np.random.randint(0, blocks, self.n_sites))
                else:
                    # autocorrelated rates
                    rate_m = "autocorrelated"
                    blocks = np.min([rs.geometric(p=0.01), self.n_sites]) # mean = 100
                    sites_indices = np.sort(rs.integers(0, blocks, self.n_sites))

            # 1. Simulate a tree and get the eigenvectors
            mean_br_length = np.exp(rs.uniform(np.log(self.min_avg_br_length), np.log(self.max_avg_br_length)))
            if self.verbose:
                print("mean_br_length", mean_br_length)
                print_update("simulating tree...")
            else:
                if init_seed == self.base_seed:
                    print_update("Running simulation %s of %s " % (sim_i + 1, n_sims))
            t = simulateTree(self.n_taxa, mean_br_length)  # args are: ntips, mean branch lengths
            # x = pn.pca(t)  # x is a dict with:
            # "eigenval"-> eigenvalues; "eigenvect"-> eigenvectors; "species"-> order of labels

            # 2. Set the rates for each site and the number of sites of a given rate
            # vector specifying the number of sites under a specific rate (default: 1)
            sites_per_scale = np.unique(sites_indices, return_counts=True)[1]

            # 3. Set the model parameters (ie frequency and substitution rates)
            # scale is a vector holding the rates per site -> scaling factor to increase/decrease branch lengths
            if rate_m == "mixed_model":
                sites_indices = np.arange(self.n_sites)
                scale = []
                rate_het_model = []
                het_r = []
                if self.store_mixed_model_info:
                    for i in range(len(sites_per_scale)):
                        a, b, c = get_rnd_rates(sites_per_scale[i], rate_m=rate_m, verbose=False)
                        scale = np.concatenate([scale, a])
                        rate_het_model.append([b, sites_per_scale[i]])
                        het_r.append(c)
                else:
                    scale = np.concatenate([get_rnd_rates(b, rate_m=rate_m, verbose=False)[0] for b in sites_per_scale])
                    rate_het_model = "Mixed-model"
                    het_r = np.nan
            else:
                scale, rate_het_model, het_r = get_rnd_rates(blocks, rate_m=rate_m,
                                                             verbose=False, p=self.p_heterogeneity_model)
            scale[scale < self.min_rate] = self.min_rate
            scale_labels = scale[sites_indices]
            den = np.mean(scale_labels)
            scale_labels = scale_labels / den
            scale = scale_labels[np.unique(sites_indices, return_index=True)[1]]

            if self.subs_model_per_block:
                model_indx = rs.integers(0, len(subs_models), blocks)
                subs_model_array = subs_models[model_indx]
                freq = None
                rates = None
                ti_tv = 0.5
            else:
                dir_shape_freq = 5
                dir_shape_rate = 5
                model_indx = rs.integers(0, len(subs_models))
                subs_model_array = subs_models[np.repeat(model_indx, blocks)]
                freq = list(rs.dirichlet([dir_shape_freq] * 4))
                rates = list(rs.dirichlet([dir_shape_rate] * 6))
                ti_tv = rs.uniform(2, 12)

            # simulate the data
            if self.verbose:
                print_update("simulating tree...done\nsimulating data...")
            aln = simulateDNA(t,
                              sites_per_scale[0],
                              scale=scale[0],
                              subs_model=subs_model_array[0],
                              freq=freq,
                              rates = rates,
                              ti_tv = ti_tv,
                              seqgen_path=self.seqgen_path)
            for i in range(1, len(sites_per_scale)):
                d = simulateDNA(t, sites_per_scale[i],
                                scale=scale[i],
                                subs_model=subs_model_array[i],
                                freq=freq,
                                rates=rates,
                                ti_tv=ti_tv,
                                seqgen_path=self.seqgen_path)
                aln.char_matrices[0].extend_matrix(d.char_matrices[0])

            if self.verbose:
                print_update("simulating data...done\nextracting features...")
            # convert to pandas dataframe
            ali = df_from_charmatrix(aln.char_matrices[0], categorical=False)

            # one hot encoding
            l = [
                ["A"] * self.n_sites,
                ["C"] * self.n_sites,
                ["G"] * self.n_sites,
                ["T"] * self.n_sites,
            ]

            ali_tmp = pd.concat([ali, pd.DataFrame(l)])
            # onehot shape = (n_taxa, n_sites * 4)
            onehot = pd.get_dummies(ali_tmp).to_numpy()[:-len(l)]
            # onehot_rs_1 shape = (n_taxa, n_sites, 4)
            onehot_rs_1 = onehot.reshape(self.n_taxa, self.n_sites, len(l))
            # onehot_rs_2 shape = (n_sites, n_taxa, 4)
            onehot_rs_2 = np.transpose(onehot_rs_1, (1, 0, 2))
            # onehot_features shape = (n_sites, n_taxa * 4)
            onehot_features = onehot_rs_2.reshape(self.n_sites, self.n_taxa * len(l))

            # get tree eigenvectors
            eigenvec = pca_from_ali(aln, tree_builder=self.tree_builder)["eigenvect"] # shape = (n_taxa, n_taxa)
            eigenvec_features = eigenvec[:,range(self.n_eigen_features)].flatten()

            if self.verbose:
                print_update("extracting features...done\n")

            features_ali.append(onehot_features)
            features_tree.append(eigenvec_features)
            labels_rates.append(scale_labels)
            labels_tl.append(t.length())
            if not self.subs_model_per_block:
                labels_smodel.append(model_indx)

            r_ml_est = None
            tl_ml_est = None
            if save_ali:
                if n_sims > 1:
                    save_ali_tmp = save_ali + str(sim_i)
                else:
                    save_ali_tmp = save_ali
                t.write_to_path(save_ali_tmp + '_true.tre', schema="newick")
                if self.ali_schema == "nexus":
                    save_ali_tmp = save_ali_tmp + '.nex'
                aln.write(path=save_ali_tmp, schema=self.ali_schema)
                if run_phyml_estimation:
                    r_ml_est, tl_ml_est = run_phyml(save_ali_tmp, path_phyml=self.phyml_path,
                                                    model=model_indx, n_sites=self.n_sites,
                                                    ncat=4)
            else:
                save_ali_tmp = ""

            info.append({
                "n_blocks": blocks,
                "mean_br_length": mean_br_length,
                "rate_het_model": [rate_het_model, het_r, rate_m],
                "model_indx": model_indx,
                "freq": freq,
                "rates": rates,
                "ti_tv": ti_tv,
                "r_ml_est": r_ml_est,
                "tl_ml_est": tl_ml_est,
                "ali_file": save_ali_tmp
            })

        return [features_ali, features_tree, labels_rates, labels_smodel, labels_tl, info]


