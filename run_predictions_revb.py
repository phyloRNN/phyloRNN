import os
import numpy as np
import phyloRNN as pn
from matplotlib import pyplot as plt
wd = "/Users/dsilvestro/Desktop/revbayes_test/trained_model"
data_wd = "/Users/dsilvestro/Desktop/revbayes_test/data"
# training_file = os.path.join(os.getcwd(), "training_data.npz")
model_name = "t20_s100_model"
trained_model = pn.load_rnn_model(os.path.join(wd, model_name))

plot = False
log_rates = False
n_sim = 100

# simulate data

sim = pn.simulator(n_taxa = 20,
                   n_sites = 100,
                   n_eigen_features = 3,
                   min_rate = 0,  #
                   freq_uncorrelated_sites = 0.5,
                   freq_mixed_models = 0,
                   store_mixed_model_info = True,
                   tree_builder = 'nj',  # 'upgma'
                   subs_model_per_block = False,  # if false same subs model for all blocks
                   phyml_path = None, #os.path.join(os.getcwd(), "phyloRNN"),
                   seqgen_path = None, # os.path.join(os.getcwd(), "phyloRNN", "seq-gen")
                   ali_path = data_wd, #os.path.join(os.getcwd(), "phyloRNN", "ali_tmp"),
                   DEBUG=False,
                   verbose = True,
                   ali_schema = "nexus", # phylip,
                   min_avg_br_length=0.01,
                   max_avg_br_length=0.2
                   )

# run simulations set
for sim_i in range(n_sim):
    ali_name = os.path.join(data_wd, "ali%s" % sim_i)
    res = sim.run_sim([123, 1,
                 ali_name,
                 False])

    ali_file = res[-1][0]['ali_file']

    # get features for rnn predictions
    true_site_rates = res[2][0]
    sim_res = {'features_ali': res[0][0],
               'labels_rates': true_site_rates,
               'labels_smodel': None,
                'labels_tl': res[-2][0]
               }

    # create input
    (comp_sim, dict_inputs, comp_dict_outputs
     ) = pn.rnn_in_out_dictionaries_from_sim(sim=sim_res,
                                          log_rates=log_rates,
                                          output_list=['per_site_rate','tree_len'],
                                          include_tree_features=False)



    print("Running predictions...")
    model_input = {'sequence_data': dict_inputs['sequence_data'].reshape((1,
                                          dict_inputs['sequence_data'].shape[0],
                                          dict_inputs['sequence_data'].shape[1]))
                   }
    predictions = trained_model.predict(model_input)

    site_rates = predictions[0][0]
    if plot:
        print("MSE:", np.mean((true_site_rates - site_rates)**2))
        plt.scatter(true_site_rates, site_rates)
        plt.show()
        print(res[-1][0]['rate_het_model'])

    x = np.linspace(0, 1, 5)
    qntl = x[1:] - (x[1] / 2)
    discrete_gamma = False
    if discrete_gamma:
        site_rates_discrete = np.quantile(site_rates, qntl)
    else:
        site_rates_discrete = site_rates
    # print(site_rates_discrete)
    print("tree length:", )

    pn.get_revBayes_script(ali_file, ali_name, ali_name,
                           sr=None, gamma_model=True, inv_model=True)

    pn.get_revBayes_script(ali_file, ali_name, ali_name,
                           sr=site_rates_discrete, gamma_model=False, partitioned=True)


    pn.save_pkl(res, ali_name + "_info.pkl")

