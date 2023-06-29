import os
import numpy as np
import phyloRNN as pn
from tensorflow import keras
np.set_printoptions(suppress=True, precision=3)

sim_file = "/home/silvestr/Documents/phyloRNN/training_data20230112_e3_nj.npz"
wd = os.path.dirname(sim_file)


# load data
sim, dict_inputs, dict_outputs = pn.rnn_in_out_dictionaries_from_sim(sim_file,
                                                                     log_rates=False,
                                                                     log_tree_len=True,
                                                                     output_list=['per_site_rate','tree_len'],
                                                                     include_tree_features=False,)

n_onehot_classes = 4  # a, c, g, t
(n_instances, n_sites, n) = sim['features_ali'].shape
n_taxa = int(n / n_onehot_classes)
# tree_len_rescaler = n_sites / n_taxa
# dict_outputs["tree_len"] =  dict_outputs["tree_len"] / tree_len_rescaler
# dict_outputs["tree_len"] =  np.sqrt(dict_outputs["tree_len"])

# build model
node_list = [128, # 0. sequence_LSTM_1
             64,  # 1. sequence_LSTM_2
             0,  # 2. phylo_FC_1 (dense on phylo features)
             64, # 3. site_NN and site_NN_tl (block-NNs dense with shared prms)
             32, # 4. site_rate_hidden
             0,  # 5. sub_model_hidden
             8, # 6. tree_len_hidden
             0   # 7. tree_len_hidden_2 (if > 0)
             ]
model_name = "sepBLOCKresH128-8logTL"

model = pn.build_rnn_model(n_sites=n_sites,
                           n_species=n_taxa,
                           n_eigenvec=0,
                           bidirectional_lstm=True,
                           loss_weights=[1, 1, 1],
                           nodes=node_list,
                           pool_per_site=True,
                           output_list=['per_site_rate','tree_len'],
                           mean_normalize_rates=True,
                           layers_norm=False,
                           separate_block_nn=True,
                           output_f = ['softplus','softmax','linear'],
                           optimizer=keras.optimizers.RMSprop(1e-3))

# model.summary()
print("N. model parameters:", model.count_params())

# training
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss",
                                           patience=20,
                                           restore_best_weights=True)

history = model.fit(dict_inputs, dict_outputs,
                    epochs=1000,
                    validation_split=0.2,
                    verbose=1,
                    callbacks=[early_stop],
                    batch_size=100)

# save model
pn.save_rnn_model(wd=wd,
                  history=history,
                  model=model, feature_rescaler=None, filename=model_name)




