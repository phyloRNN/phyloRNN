# Import the memory profiler
from memory_profiler import profile
import phyloRNN as pn


@profile
def train():
    # load data
    sim, dict_inputs, dict_outputs = pn.rnn_in_out_dictionaries_from_sim("training_data_taxa_50_sites1000_20000.db",
                                                                         log_rates=False,
                                                                         log_tree_len=True,
                                                                         output_list=['per_site_rate', 'tree_len'],
                                                                         include_tree_features=False,
                                                                         sqlite=True)
    # setup model architecture
    model_config = pn.rnn_config(n_sites=1000, n_taxa=50)  # default settings

    # build model
    model = pn.build_rnn_model(model_config,
                               optimizer=pn.keras.optimizers.RMSprop(1e-3),
                               print_summary=False)

    # training
    early_stop = pn.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                  patience=5,
                                                  restore_best_weights=True)

    model.fit(dict_inputs, dict_outputs,
                        epochs=20,
                        verbose=2,
                        #callbacks=[early_stop],
                        batch_size=500)

'''
To run:
    mprof run filename.py
    mprof plot 
or 
    python -m memory_profiler normal_training.py

'''
if __name__ == '__main__':
    train()
