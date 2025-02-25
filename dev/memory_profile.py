# Import the memory profiler
from memory_profiler import profile
import phyloRNN as pn

@profile
def train(fn, sqlite, taxa, site):
    # load data
    sim, dict_inputs, dict_outputs = pn.rnn_in_out_dictionaries_from_sim(fn,
                                                                         log_rates=False,
                                                                         log_tree_len=True,
                                                                         output_list=['per_site_rate', 'tree_len'],
                                                                         include_tree_features=False,
                                                                         sqlite=sqlite)
    # setup model architecture
    model_config = pn.rnn_config(n_sites=site, n_taxa=taxa)  # default settings

    # build model
    model = pn.build_rnn_model(model_config,
                               optimizer=pn.keras.optimizers.RMSprop(1e-3),
                               print_summary=False)

    # training
    early_stop = pn.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                  patience=5,
                                                  restore_best_weights=True)

    model.fit(dict_inputs, dict_outputs,
                        epochs=5,
                        validation_split=0.2,
                        verbose=2,
                        callbacks=[early_stop],
                        batch_size=10)

'''
To run:
    mprof run filename.py
    mprof plot 
or 
    python -m memory_profiler memory_profile.py

'''
if __name__ == '__main__':
    train('../data/training_data_taxa_20_sites200_2000.db', True, 20, 200)
    #train('../notebooks/training_data_taxa_10_sites100_200.db', True, 10, 100)