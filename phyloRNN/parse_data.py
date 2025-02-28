import dendropy
from Bio import AlignIO
import re
from .rate_estimator import *
from .sequence_simulator import pca_from_ali
from .utilities import *
import tensorflow as tf
import sqlite3
import zlib
import numpy as np
import json


def _decompdecode(e, dtype, shape=False):
    if shape:
        return np.frombuffer(zlib.decompress(e), dtype=dtype).reshape(shape)
    else:
        return np.frombuffer(zlib.decompress(e), dtype=dtype)

def generate_features(wd="data", n_eigen=3, output=""):
    feature_set = []
    label_set = []
    i = 1
    while True:
        # read files
        try:
            ali = AlignIO.read(os.path.join(wd, "sim_msa_%s.phy" % i),
                               format="phylip-relaxed")
            print_update(str(i))
        except:
            break
        eig_tbl = pd.read_csv(os.path.join(wd, "sim_eigenvectors_tree_%s.csv" % i),
                              header=None)

        # prep features
        ali_features = onehot_encoder(ali)
        eigen_features = get_eigen_vectors(eig_tbl, ali, size=n_eigen)

        # parse labels
        ali_labels = pd.read_csv(os.path.join(wd, "sim_rates_%s.csv" % i),
                                 header=None).to_numpy()[:, 1].flatten()

        """
        f.shape = (n. sites, n. features), n. features = 
            = [4 (acgt) + sp. eigenvector] x n. species
        """

        f = flatten_features(ali_features, eigen_features)
        feature_set.append(f)
        label_set.append(ali_labels)
        i += 1

    if len(label_set):
        feature_set = np.array(feature_set).astype(float)
        label_set = np.array(label_set)

        np.save("%sfeature_set.npy" % output, feature_set)
        np.save("%slabel_set.npy" % output, label_set)
        print("\nFeatures saved as: %slabel_set.npy" % output, "%sfeature_set.npy" % output)
    else:
        print("didnt work")

def onehot_encoder(ali):
    ntaxa = len(ali._records)
    coded_ali = []
    for i in range(ntaxa):
        a = str(ali.__getitem__(i).seq)
        n = list(a)
        if len(np.unique(n)) != 4:
            print("\n", len(np.unique(n)), "nucleotides found")
            n_tmp = n + ['a','t','c','g']
            # print(np.unique(n_tmp), len(n), len(n_tmp[4:]))
            onehot = pd.get_dummies(n_tmp).to_numpy()
            onehot = onehot[4:, :]
            # print(i, onehot[4:,:].shape)
        else:
            onehot = pd.get_dummies(n).to_numpy()
            # print(i, onehot.shape)
        coded_ali.append(onehot)
    coded_ali = np.array(coded_ali)
    return coded_ali

def flatten_features(ali_onehot, eigen_features):
    # init shape
    "ali_onehot.shape = (n_species, n_sites, one_hot)"
    f = np.transpose(ali_onehot, (1, 0, 2))  # sites, species, one-hot
    (n_sites, n_species, one_hot) = f.shape
    # concatenate eigenvectors
    d = np.broadcast_to(eigen_features, (n_sites, n_species, eigen_features.shape[1]))
    f_reshape = f.reshape(n_sites, n_species * one_hot)
    d_reshape = d.reshape(n_sites, n_species * eigen_features.shape[1])
    return np.concatenate((f_reshape, d_reshape), axis=1)

def concatenate_features(ali_onehot, eigen_features):
    # init shape
    "ali_onehot.shape = (n_species, n_sites, one_hot)"
    f = np.transpose(ali_onehot, (1, 0, 2))  # sites, species, one-hot
    (n_sites, n_species, one_hot) = f.shape
    # concatenate eigenvectors
    d = np.broadcast_to(eigen_features, (n_sites, n_species, eigen_features.shape[1]))
    pass



def flatten_features_ali(ali_onehot, eigen_vec):
    # init shape
    (n_species, n_sites, one_hot) = ali_onehot.shape
    # f = np.transpose(ali_onehot, (0, 2, 1))  # sites, species, one-hot
    # (n_sites, n_species, one_hot) = f.shape
    # TODO: concatenate eigenvectors
    ali_onehot_reshaped = ali_onehot.reshape(n_species, n_sites * one_hot)




def get_eigen_vectors(eig_tbl, ali, size=None):
    "return eigen vectors in the same order as taxa in ali"
    ntaxa = len(ali._records)
    eig_taxa = eig_tbl.to_numpy()[:,0]
    eig_values = eig_tbl.to_numpy()[:, 1:]
    if size is None:
        size = eig_values.shape[0]
    eigen_vec = []
    for i in range(ntaxa):
        indx = np.where(eig_taxa == ali.__getitem__(i).name)[0][0]
        eigen_vec.append(eig_values[indx,:size])

    return np.array(eigen_vec)

def np_to_tf(x, type=np.float32):
    with tf.device('/cpu:0'):
        tf_x = tf.convert_to_tensor(np.array(x), type)
    return tf_x


def sqlite_data_generator(db_path, batch_size):

    """Generator that loads data from an SQLite database in batches."""
    conn = sqlite3.connect(db_path)  # Connect to the database
    cursor = conn.cursor()

    # Retrieve data from array_info table
    array_info_dict = {}
    cursor.execute('SELECT name_, dtype, shape FROM array_info')

    for row in cursor.fetchall():
        name_, dtype, shape = row
        array_info_dict[name_] = {
            'dtype': dtype,
            'shape': eval(shape)[1:]  # Remove the first dimension because its just the number of simulation
        }

    cursor.execute("SELECT COUNT(*) FROM simulation")
    total_samples = cursor.fetchone()[0]  # Get total number of rows

    for offset in range(0, total_samples, batch_size):

        # Query to get the data
        query = "SELECT features_ali, features_tree, labels_rates, labels_smodel, labels_tl, info FROM simulation  LIMIT ? OFFSET ?"
        cursor.execute(query, (batch_size, offset))
        rows = cursor.fetchall()

        if not rows:
            break  # Stop if no more data

        sim = {
            'features_ali': [],
            'features_tree': [],
            'labels_rates': [],
            'labels_smodel': [],
            'labels_tl': [],
            'info': []
        }

        for row in rows:
            sim['features_ali'].append(_decompdecode(row[0], array_info_dict['features_ali']['dtype'],
                                                     array_info_dict['features_ali']['shape']))
            sim['features_tree'].append(_decompdecode(row[1], array_info_dict['features_tree']['dtype']))
            sim['labels_rates'].append(
                _decompdecode(row[2], array_info_dict['labels_rates']['dtype']) if row[2] else None)
            sim['labels_smodel'].append(_decompdecode(row[3], array_info_dict['labels_smodel']['dtype'],
                                                      array_info_dict['labels_smodel']['shape']) if row[3] else None)
            sim['labels_tl'].append(
                _decompdecode(row[4], array_info_dict['labels_tl']['dtype'], array_info_dict['labels_tl']['shape']) if
                row[4] else None)
            sim['info'].append(np.array(json.loads(row[5])))

        # Tod this reshape could be avoided if we store the labels as a value instead single values array
        sim['labels_smodel'] = np.array(sim['labels_smodel']).reshape(len(sim['labels_smodel']))
        sim['labels_tl'] = np.array(sim['labels_tl']).reshape(len(sim['labels_tl']))

        sim, dict_inputs, dict_outputs = rnn_in_out_dictionaries_from_sim(sim=sim, log_rates=False,
                                                                             log_tree_len=True,
                                                                             output_list=['per_site_rate', 'tree_len'],
                                                                             include_tree_features=False, sqlite=True)

        yield dict_inputs, dict_outputs

    conn.close()

def rnn_in_out_dictionaries_from_sim(sim_file=None,
                                     sim=None,
                                     log_rates=True,
                                     log_tree_len=True,
                                     output_list=None,
                                     include_tree_features=True,
                                     sqlite=False):
    if sim is None:
        if sqlite:

            # Connect to the SQLite database
            conn = sqlite3.connect(sim_file)
            cursor = conn.cursor()

            # Retrieve data from array_info table
            array_info_dict = {}
            cursor.execute('SELECT name_, dtype, shape FROM array_info')

            for row in cursor.fetchall():
                name_, dtype, shape = row
                array_info_dict[name_] = {
                    'dtype': dtype,
                    'shape': eval(shape)[1:] # Remove the first dimension because its just the number of simulation
                }

            # Query to get the data
            cursor.execute(
                "SELECT features_ali, features_tree, labels_rates, labels_smodel, labels_tl, info FROM simulation")
            rows = cursor.fetchall()

            sim = {
                'features_ali': [],
                'features_tree': [],
                'labels_rates': [],
                'labels_smodel': [],
                'labels_tl': [],
                'info': []
            }

            for row in rows:
                sim['features_ali'].append(_decompdecode(row[0],array_info_dict['features_ali']['dtype'], array_info_dict['features_ali']['shape']))
                sim['features_tree'].append(_decompdecode(row[1],array_info_dict['features_tree']['dtype']))
                sim['labels_rates'].append(_decompdecode(row[2],array_info_dict['labels_rates']['dtype']) if row[2] else None)
                sim['labels_smodel'].append(_decompdecode(row[3],array_info_dict['labels_smodel']['dtype'], array_info_dict['labels_smodel']['shape']) if row[3] else None)
                sim['labels_tl'].append(_decompdecode(row[4],array_info_dict['labels_tl']['dtype'], array_info_dict['labels_tl']['shape']) if row[4] else None)
                sim['info'].append(np.array(json.loads(row[5])))

            # Tod this reshape could be avoided if we store the labels as a value instead single values array
            sim['labels_smodel'] = np.array(sim['labels_smodel']).reshape(len(sim['labels_smodel']))
            sim['labels_tl'] = np.array(sim['labels_tl']).reshape(len(sim['labels_tl']))


            # Close the database connection
            conn.close()
        else:
            sim = dict(np.load(sim_file, allow_pickle=True))

    if log_rates and sim['labels_rates'] is not None:
        sim['labels_rates'] = np.log10(sim['labels_rates'])
    if log_tree_len:
        def f(x):
            return np.log10(x)
    else:
        def f(x):
            return x

    if output_list is None:
        output_list = ['per_site_rate', 'sub_model', 'tree_len']

    # prep data
    if include_tree_features:
        dict_inputs = {
            "sequence_data": np_to_tf(sim['features_ali']),
            "eigen_vectors": np_to_tf(sim['features_tree'])
        }
    else:
        dict_inputs = {
            "sequence_data": np_to_tf(sim['features_ali']),
        }

    dict_outputs = {}
    if 'per_site_rate' in output_list:
        dict_outputs["per_site_rate"] = np_to_tf(sim['labels_rates'])
    if 'sub_model' in output_list and sim['labels_smodel'] is not None:
        dict_outputs["sub_model"] = np_to_tf(pd.get_dummies(sim['labels_smodel']).to_numpy())
    if "tree_len" in output_list and sim['labels_tl'] is not None:
        dict_outputs["tree_len"] = np_to_tf(f(sim['labels_tl']))

    return sim, dict_inputs, dict_outputs


def randomize_sites(dict_inputs, dict_outputs):
    # dict_inputs['sequence_data'].shape = (simulations, sites, species * 4)
    # dict_inputs['eigen_vectors'].shape = (simulations, species)
    # dict_outputs['per_site_rate'].shape = (simulations, sites)
    n_sites = dict_inputs['sequence_data'].shape[1]
    rnd_indx = np.random.choice(range(n_sites), size=n_sites, replace=False)
    rnd_dict_inputs = copy.deepcopy(dict_inputs)
    rnd_dict_outputs = copy.deepcopy(dict_outputs)
    rnd_dict_inputs['sequence_data'] = rnd_dict_inputs['sequence_data'][:,rnd_indx,:]
    rnd_dict_outputs['per_site_rate'] = rnd_dict_outputs['per_site_rate'][:,rnd_indx]
    return rnd_dict_inputs, rnd_dict_outputs, rnd_indx




def gap_fixer(ali, dict=['A','C','G','T']):
    np_ali = np.array(ali)
    (base, count) = np.unique(np_ali, return_counts=True)
    most_common_overall = base[np.argmax(count)]

    for site_i in range(np_ali.shape[1]):
        s = np_ali[:,site_i]
        (base, count) = np.unique(s, return_counts=True)
        for i in range(len(count)):
            if base[i] not in dict:
                count[i] *= 0
        if np.sum(count) > 0:
            most_common = base[np.argmax(count)]
        else: # if they are all gaps
            most_common = most_common_overall

        for i in range(len(count)):
            if base[i] not in dict:
                s[s == base[i]] = most_common

    return pd.DataFrame(np_ali)



def parse_alignment_file(ali_file,
                         schema="fasta",
                         log_rates=False,
                         log_tree_len=True,
                         output_list=None,
                         include_tree_features=False,
                         run_phyml_estimation = False,
                         phyml_path="",
                         save_nogap_ali=False,
                         n_eigen_features = 0
                         ):
    # read fasta file
    dna = dendropy.DnaCharacterMatrix.get(file=open(ali_file), schema=schema)
    if phyml_path == "":
        phyml_path = os.path.join(os.getcwd(), "phylogeNN")

    # convert to pandas dataframe
    ali = df_from_charmatrix(dna, categorical=False)

    (n_taxa, n_sites) = ali.shape
    # one hot encoding
    l = [
        ["A"] * n_sites,
        ["C"] * n_sites,
        ["G"] * n_sites,
        ["T"] * n_sites,
    ]
    ali_tmp = pd.concat([ali, pd.DataFrame(l)])

    ali_tmp_nogap = gap_fixer(ali_tmp)
    if save_nogap_ali:
        ali_tmp_nogap.to_csv(ali_file + "_imputed.phy")

    if len(np.unique(ali_tmp_nogap)) != 4:
        ValueError("Error - only A, C, G, T allowed! Found:", np.unique(ali_tmp_nogap))
    # onehot shape = (n_taxa, n_sites * 4)
    onehot = pd.get_dummies(ali_tmp_nogap).to_numpy()[:-len(l)]
    # onehot_rs_1 shape = (n_taxa, n_sites, 4)
    onehot_rs_1 = onehot.reshape(n_taxa, n_sites, len(l))
    # onehot_rs_2 shape = (n_sites, n_taxa, 4)
    onehot_rs_2 = np.transpose(onehot_rs_1, (1, 0, 2))
    # onehot_features shape = (n_sites, n_taxa * 4)
    onehot_features = onehot_rs_2.reshape(n_sites, n_taxa * len(l))

    # get tree eigenvectors
    if include_tree_features:
        eigenvec = pca_from_ali(dna, tree_builder='nj')["eigenvect"]  # shape = (n_taxa, n_taxa)
        eigenvec_features = eigenvec[:, range(n_eigen_features)].flatten()
    else:
        eigenvec_features = None
    features_ali = []
    features_tree = []
    info = []

    features_ali.append(onehot_features)
    features_tree.append(eigenvec_features)

    if run_phyml_estimation:
        dna.write(path=ali_file + ".phy", schema="phylip")
        r_ml_est, tl_ml_est = run_phyml(ali_file + ".phy", path_phyml=phyml_path, model=2, ncat=4)
    else:
        r_ml_est = None
        tl_ml_est = None

    info.append({
        "r_ml_est": r_ml_est,
        "tl_ml_est": tl_ml_est
    })

    sim = {'features_ali': np.array(features_ali),
                            'features_tree' : np.array(features_tree),
                            'labels_rates' : None,
                            'labels_smodel' : None,
                            'labels_tl' : None,
                            'info' : np.array(info)
    }



    dat, dict_inputs, dict_outputs = rnn_in_out_dictionaries_from_sim(
        sim_file=None,
        sim=sim,
        log_rates=log_rates,
        log_tree_len=log_tree_len,
        output_list=output_list,
        include_tree_features=include_tree_features)

    return dat, dict_inputs, dict_outputs



def parse_large_alignment_file(ali_file,
                               batch_size,
                               n_taxa,
                               truncate=None):
    counter = 1
    features = []
    with open(ali_file, "r") as f:
        ind = 0
        for line in f.readlines():
            if ind:
                print_update("Taxon", counter)
                tmp = re.sub("A", "1000", line)
                tmp = re.sub("C", "0100", tmp)
                tmp = re.sub("G", "0010", tmp)
                tmp = re.sub("T", "0001", tmp)
                tmp_list = [*tmp]
                tmp_list = tmp_list[:-1]  # remove last character: '\n'
                # print(np.array(tmp_list).shape)
                if truncate is not None:
                    tmp_list = tmp_list[:(truncate * 4)]
                features.append(tmp_list)
                ind = 0
                counter += 1
            else:
                ind = 1

    ##
    tot_n_sites = len(features[0]) / 4
    print("\ntot_n_sites:", tot_n_sites)
    predict_sites = round((len(features[0]) / 4) / batch_size) * batch_size
    features_ali = []
    for i in range(0, predict_sites, batch_size):
        if len(features_ali) % 50 == 0:
            print_update("Sites: %s / %s (%s)" % (i, predict_sites, np.round((100 * i / predict_sites), 2)))
        feature_sites = [tmp[i * 4: (i + batch_size) * 4] for tmp in features]
        onehot = np.array(feature_sites).astype(int)
        # onehot_rs_1 shape = (n_taxa, n_sites, 4)
        onehot_rs_1 = onehot.reshape(n_taxa, batch_size, 4)
        # onehot_rs_2 shape = (n_sites, n_taxa, 4)
        onehot_rs_2 = np.transpose(onehot_rs_1, (1, 0, 2))
        # onehot_features shape = (n_sites, n_taxa * 4)
        onehot_features = onehot_rs_2.reshape(batch_size, n_taxa * 4)
        features_ali.append(onehot_features)

    dict_inputs = {"sequence_data": np.array(features_ali), }
    print("\ndone.\n")
    return dict_inputs






















