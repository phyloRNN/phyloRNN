import os
import phyloRNN as pn

training_file = os.path.join(os.getcwd(), "training_data.npz")
wd = os.path.dirname(training_file)
model_name = "phylo_rnn_model"

# load data
sim, dict_inputs, dict_outputs = pn.rnn_in_out_dictionaries_from_sim(training_file,
                                                                     log_rates=False,
                                                                     log_tree_len=True,
                                                                     output_list=['per_site_rate','tree_len'],
                                                                     include_tree_features=False)

# setup model architecture
model_config = pn.rnn_config(n_sites=1000, n_taxa=50) # default settings

# build model
model = pn.build_rnn_model(model_config,
                           optimizer=pn.keras.optimizers.RMSprop(1e-3),
                           print_summary=False)


# training
early_stop = pn.keras.callbacks.EarlyStopping(monitor="val_loss",
                                              patience=5,
                                              restore_best_weights=True)

history = model.fit(dict_inputs, dict_outputs,
                    epochs=1000,
                    validation_split=0.2,
                    verbose=1,
                    callbacks=[early_stop],
                    batch_size=100)

# save model
pn.save_rnn_model(wd=wd, history=history, model=model, filename=model_name)

