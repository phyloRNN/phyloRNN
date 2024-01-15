import pandas as pd

from .plots import *
from .utilities import *
from .parse_data import *
from .rnn_builder import *
import matplotlib.pyplot as plt


def compare_rnn_phyml(compare_file, # *.npz
                      model_file,
                      output_dir,
                      model_wd=None,
                      log_rates = False,
                      log_tl = True,
                      tree_features = False,
                      include_tree_features = False,
                      n_taxa = 50,
                      output_list = None,
                      parse_heterogeneity_models=False,
                      plot_name="rnn_res",
                      plot_rate_indices=[0,1,2], # plot first three sims
                      plot_n_sim=50,
                      rnn_model_tag="rnn_model",
                      sqrt_tl=False,
                      rescale_tl=1, # legacy default: (0.5 * n_taxa)
                      plot_results=True
                      ):
    if output_list is None:
        output_list = ['per_site_rate', 'tree_len']

    print("Parsing data...")
    (comp_sim, comp_dict_inputs, comp_dict_outputs
     ) = rnn_in_out_dictionaries_from_sim(compare_file,
                                          log_rates=log_rates,
                                          output_list=output_list,
                                          include_tree_features=include_tree_features)

    if tree_features == 0 and include_tree_features == 1:
        comp_sim['features_tree'] = np.zeros((comp_sim['features_tree'].shape[0], n_taxa))
        n_eigenvec = int(comp_sim['features_tree'].shape[1] / n_taxa)
        comp_dict_inputs['eigen_vectors'] = comp_sim['features_tree']

    all_res = {}
    # --- load model
    # for model_i in range(len(model_names)):
    print("Loading rnn model...")
    m = load_rnn_model(wd=model_wd, filename=rnn_model_tag + model_file)

    # run predictions
    print("Running predictions...")
    comp_predictions = m.predict(comp_dict_inputs)

    # get phyML rate estimates
    mle_fr = np.array(
        [comp_sim['info'][i]['r_ml_est']['rate_fr'].to_numpy() for i in range(len(comp_sim['info']))]).astype(float)
    mle_g = np.array(
        [comp_sim['info'][i]['r_ml_est']['rate_g'].to_numpy() for i in range(len(comp_sim['info']))]).astype(float)

    if log_rates:
        y_true = 10 ** comp_sim['labels_rates']
        dle = 10 ** comp_predictions[0]
    else:
        y_true = comp_sim['labels_rates']
        dle = comp_predictions[0]

    print("Per-site rates - All simulations")
    mle_g += np.random.uniform(0, 0.000001, mle_fr.shape)
    mle_fr += np.random.uniform(0, 0.000001, mle_g.shape)
    all_res['rates-all'] = print_msa_compare(mle_fr, mle_g, dle, y_true)

    if plot_results:
        plot_sample_rate_results(dle, mle_g, mle_fr, y_true,
                                    n_sim_plot=plot_n_sim,
                                    sim_indx=np.arange(plot_n_sim),
                                    rate_indices=plot_rate_indices, # plot first three
                                    wd=output_dir,
                                    outname=plot_name + "_rates"
                                    )


    # collect tree lengths
    tl_mle = pd.concat([comp_sim['info'][i]['tl_ml_est'] for i in range(len(comp_sim['info']))])
    tl_mle_fr = tl_mle['tl_fr'].to_numpy()
    tl_mle_g = tl_mle['tl_g'].to_numpy()
    if log_tl:
        tl_dle = 10 ** comp_predictions[1].flatten()
    else:
        if sqrt_tl:
            tl_dle = comp_predictions[1].flatten() ** 2
        elif rescale_tl:
            tl_dle = rescale_tl * comp_predictions[1].flatten()
        else:
            tl_dle = comp_predictions[1].flatten()
    tl_true = np.array(comp_sim['labels_tl'])

    print("Tree Length - All simulations (nrmse)")
    print(get_nrmse(tl_mle_g[np.isfinite(tl_mle_g)], tl_true[np.isfinite(tl_mle_g)]))
    print(get_nrmse(tl_mle_fr[np.isfinite(tl_mle_fr)], tl_true[np.isfinite(tl_mle_fr)]))
    print(get_nrmse(tl_dle, tl_true))

    all_res['tl-all'] = print_mape_compare(tl_mle_fr, tl_mle_g, tl_dle, tl_true)

    if plot_results:
        plot_treelength(tl_mle_fr, tl_mle_g, tl_dle, tl_true,
                        indx=None, alpha=0.5,  # het_indx_list
                        show=False, wd=output_dir,
                        outname=plot_name + "_tl")

    # plot by rate heterogeneity
    if parse_heterogeneity_models:
        print("\nResults by rate heterogeneity model")
        hm = []
        for i in range(len(comp_sim['info'])):
            try:
                hm_str = comp_sim['info'][i]['rate_het_model'][0] + "_" + comp_sim['info'][i]['rate_het_model'][2]
            except:
                hm_str = comp_sim['info'][i]['rate_het_model'][-1] + "_" + comp_sim['info'][i]['rate_het_model'][2]
            hm.append(hm_str)
        hm = np.array(hm)

        het_names = np.unique(hm)
        het_indx_list = [np.where(hm == i)[0] for i in het_names]

        # run for each heterogeneity model
        for i in range(len(het_names)):
            print("\n%s" % het_names[i])
            r = print_msa_compare(mle_fr, mle_g, dle, y_true, het_indx_list[i])
            all_res['rates-' + het_names[i]] = r
            if plot_results:
                plot_sample_rate_results(dle, mle_g, mle_fr, y_true,
                                         n_sim_plot=10,
                                         sim_indx=het_indx_list[i][0:10],
                                         rate_indices=het_indx_list[i][0:3],
                                         wd=output_dir,
                                         outname=het_names[i],
                                         mse_values=r['mse'])

        for i in range(len(het_names)):
            print("\n Tree Length:", het_names[i])
            all_res['tl-' + het_names[i]] = print_mape_compare(tl_mle_fr, tl_mle_g, tl_dle, tl_true, het_indx_list[i])

    return m, comp_predictions, comp_sim, comp_dict_inputs, comp_dict_outputs, all_res



def compare_rnn_models(model_names, model_wd, log=True):
    # --- plot training results
    res_training = []
    for i in range(len(model_names)):
        h = load_pkl(os.path.join(model_wd, "%s_history.pkl" % model_names[i]))
        d = plot_training_history(h, wd=model_wd,
                                  filename=model_names[i] + "_history",
                                  show=False,
                                  digits=4,
                                  log=log)
        d['model'] = model_names[i]
        res_training.append(d)

    pd.DataFrame(res_training).to_csv(os.path.join(model_wd, "training_results.txt"), sep='\t')
    print("Results saved here: %s" % os.path.join(model_wd, "training_results.txt"))
