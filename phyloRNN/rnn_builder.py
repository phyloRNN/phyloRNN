import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import metrics
np.set_printoptions(suppress=True, precision=3)
import os
import pickle as pkl
import scipy.stats
from .parse_data import sqlite_data_generator

from keras.layers import Activation
from tensorflow.python.keras.utils import generic_utils
class rnn_config():
    def __init__(self,
                 lstm_nodes: list = None,
                 nn_phylo_features: int = 0,
                 nn_shared_rate_tl: int = 64,
                 nn_rate: list = None,
                 nn_tl: list = None,
                 nn_sub_model: list = None,
                 n_eigenvec: int = 0,
                 bidirectional_lstm = True,
                 loss_weights: list = None,
                 pool_per_site = True,
                 output_list: list = None,
                 mean_normalize_rates = True,
                 layers_normalization = False,
                 separate_block_nn = True,
                 output_f: list = None,
                 n_sites: int = 1000,
                 n_taxa: int = 50,
                 sub_models: list = None,
                 ):

        if lstm_nodes is None:
            lstm_nodes = [128, 64]
        if nn_rate is None:
            nn_rate = [32]
        if nn_tl is None:
            nn_tl = [8]
        if nn_sub_model is None:
            nn_sub_model = [0]
        if loss_weights is None:
            loss_weights = [1, 1, 1]
        if output_list is None:
            output_list = ['per_site_rate', 'tree_len']
        if output_f is None:
            output_f = ['softplus', 'softmax', 'linear']
        if sub_models is None:
            sub_models = [0, 1, 2]  # ['JC', 'HKY', 'GTR']

        self.lstm_nodes = lstm_nodes
        self.nn_phylo_features = nn_phylo_features
        self.nn_shared_rate_tl = nn_shared_rate_tl
        self.nn_rate = nn_rate
        self.nn_sub_model = nn_sub_model
        self.nn_tl = nn_tl
        self.n_eigenvec = n_eigenvec
        self.bidirectional_lstm = bidirectional_lstm
        self.loss_weights = loss_weights
        self.pool_per_site = pool_per_site
        self.output_list = output_list
        self.mean_normalize_rates = mean_normalize_rates
        self.layers_norm = layers_normalization
        self.separate_block_nn = separate_block_nn
        self.output_f = output_f

        self.n_sites = n_sites
        self.n_species = n_taxa
        self.n_onehot = 4
        self.sub_models = sub_models


def build_nn_prm_share_rnn(model_config: rnn_config,
                           optimizer=keras.optimizers.RMSprop(1e-3)
                           ):
    ali_input = keras.Input(shape=(model_config.n_sites, model_config.n_species * model_config.n_onehot,), name="sequence_data")
    if model_config.n_eigenvec:
        phy_input = keras.Input(shape=(model_config.n_species * model_config.n_eigenvec,), name="eigen_vectors")
        inputs = [ali_input, phy_input]
    else:
        inputs = [ali_input]
        phy_input = None

    # lstm on sequence data
    if not model_config.bidirectional_lstm:
        ali_rnn_1 = layers.LSTM(model_config.lstm_nodes[0], return_sequences=False, activation='tanh',
                                recurrent_activation='sigmoid', name="sequence_LSTM_1")(ali_input)
        ali_rnn_2 = None
        if len(model_config.lstm_nodes) > 1:
            if model_config.lstm_nodes[1] > 0:
                ali_rnn_2 = layers.LSTM(model_config.lstm_nodes[1], return_sequences=False, activation='tanh',
                                        recurrent_activation='sigmoid', name="sequence_LSTM_2")(ali_rnn_1)
    else:
        ali_rnn_1 = layers.Bidirectional(
            layers.LSTM(model_config.lstm_nodes[0], return_sequences=False, activation='tanh',
                        recurrent_activation='sigmoid', name="sequence_LSTM_1"))(ali_input)
        ali_rnn_2 = None
        if len(model_config.lstm_nodes) > 1:
            if model_config.lstm_nodes[1] > 0:
                ali_rnn_2 = layers.Bidirectional(
                    layers.LSTM(model_config.lstm_nodes[1], return_sequences=False, activation='tanh',
                                recurrent_activation='sigmoid', name="sequence_LSTM_2"))(ali_rnn_1)

    if model_config.n_eigenvec:
        # dense on phylo data
        phy_dnn_1 = layers.Dense(model_config.nn_phylo_features, activation='relu', name="phylo_FC_1")(phy_input)
    else:
        phy_dnn_1 = None

    #--- block w shared prms


    print("Creating blocks...")
    # print(ali_rnn_1.shape)
    if model_config.n_eigenvec:
        sys.exit("not implemented")
    else:
        # NN on raw alignment data
        comb_sites = [layers.Flatten()(i) for i in tf.split(ali_input, model_config.n_sites, axis=1)]
        site_dnn_1 = layers.Dense(model_config.nn_shared_rate_tl, activation='swish', name="site_NN_1")
        site_sp_dnn_1_list = [site_dnn_1(i) for i in comb_sites]

        # second site-specific NN + RNN output
        if ali_rnn_2 is not None:
            comb_sites_rnn_out = [tf.concat((layers.Flatten()(i), layers.Flatten()(ali_rnn_2)), 1) for i in
                                  site_sp_dnn_1_list]
        else:
            comb_sites_rnn_out = [tf.concat((layers.Flatten()(i), layers.Flatten()(ali_rnn_1)), 1) for i in
                                  site_sp_dnn_1_list]



    # add second DNN layer (1 node per site)
    if model_config.pool_per_site:
        site_dnn_2 = layers.Dense(1, activation='swish', name="site_rate_hidden2")
        site_sp_dnn_2_list = [site_dnn_2(i) for i in comb_sites_rnn_out]
        concat_1 = layers.concatenate(site_sp_dnn_2_list)
    else:
        concat_1 = layers.concatenate(comb_sites_rnn_out)
    print("done")
    #---

    # Merge all available features into a single large vector via concatenation
    # concat_1 = layers.concatenate([ali_rnn_3, phy_dnn_1])
    outputs = []
    loss = {}
    loss_w = {}
    # output 1: per-site rate
    if 'per_site_rate' in model_config.output_list:
        site_rate_1 = layers.Dense(model_config.nn_rate[0], activation='swish', name="site_rate_hidden")
        site_rate_1_list = [site_rate_1(i) for i in comb_sites_rnn_out]
        rate_pred_nn = layers.Dense(1, activation=model_config.output_f[0], name="per_site_rate_split")
        rate_pred_list = [rate_pred_nn(i) for i in site_rate_1_list]
        # rate_pred =  layers.Dense(1, activation='linear', name="per_site_rate")(layers.concatenate(rate_pred_list))

        if not model_config.mean_normalize_rates:
            rate_pred = layers.Flatten(name="per_site_rate")(layers.concatenate(rate_pred_list))
        else:
            def mean_rescale(x):
                return x / tf.reduce_mean(x, axis=1, keepdims=True)

            generic_utils.get_custom_objects().update({'mean_rescale': Activation(mean_rescale)})
            rate_pred_tmp = layers.Flatten(name="per_site_rate_tmp")(layers.concatenate(rate_pred_list))
            rate_pred = layers.Activation(mean_rescale, name='per_site_rate')(rate_pred_tmp)

        outputs.append(rate_pred)
        loss['per_site_rate'] = keras.losses.MeanSquaredError()
        loss_w["per_site_rate"] = model_config.loss_weights[0]

    # output 2: model test (e.g. JC, HKY, GTR)
    if 'sub_model' in model_config.output_list:
        subst_model_1 = layers.Dense(model_config.nn_sub_model[0], activation='relu', name="sub_model_hidden")(concat_1)
        sub_model_pred = layers.Dense(len(model_config.sub_models),
                                      activation=model_config.output_f[1],
                                      name="sub_model")(subst_model_1)
        outputs.append(sub_model_pred)
        loss['sub_model'] = keras.losses.CategoricalCrossentropy(from_logits=False)
        loss_w['sub_model'] = model_config.loss_weights[1]

    # output 3: tree length
    if 'tree_len' in model_config.output_list:
        tree_len_1 = layers.Dense(model_config.nn_tl[0], activation='swish', name="tree_len_hidden")(concat_1)
        if len(model_config.nn_tl) == 1:
            tree_len_pred = layers.Dense(1, activation=model_config.output_f[2], name="tree_len")(tree_len_1)
        else:
            if model_config.nn_tl[1] > 0:
                tree_len_2 = layers.Dense(model_config.nn_tl[1], activation='swish', name="tree_len_hidden_2")(tree_len_1)
                tree_len_pred = layers.Dense(1, activation=model_config.output_f[2], name="tree_len")(tree_len_2)
            else:
                tree_len_pred = layers.Dense(1, activation=model_config.output_f[2], name="tree_len")(tree_len_1)
        outputs.append(tree_len_pred)
        loss['tree_len'] = keras.losses.MeanSquaredError()
        loss_w['tree_len'] = model_config.loss_weights[2]

    # Instantiate an end-to-end model predicting both rates and substitution model
    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        loss_weights=loss_w
    )

    return model

def build_rnn_model(model_config: rnn_config,
                    optimizer=keras.optimizers.RMSprop(1e-3),
                    print_summary=False
                    ):
    ali_input = keras.Input(shape=(model_config.n_sites, model_config.n_species * model_config.n_onehot,),
                            name="sequence_data")
    if model_config.n_eigenvec:
        phy_input = keras.Input(shape=(model_config.n_species * model_config.n_eigenvec,),
                                name="eigen_vectors")
        inputs = [ali_input, phy_input]
    else:
        inputs = [ali_input]

    # lstm on sequence data
    if not model_config.bidirectional_lstm:
        ali_rnn_1 = layers.LSTM(model_config.lstm_nodes[0], return_sequences=True, activation='tanh',
                                recurrent_activation='sigmoid', name="sequence_LSTM_1")(ali_input)
        if model_config.layers_norm:
            ali_rnn_1n = layers.LayerNormalization(name='layer_norm_rnn1')(ali_rnn_1)
        else:
            ali_rnn_1n = ali_rnn_1
        ali_rnn_2 = layers.LSTM(model_config.lstm_nodes[1], return_sequences=True, activation='tanh',
                                recurrent_activation='sigmoid', name="sequence_LSTM_2")(ali_rnn_1n)
        if model_config.layers_norm:
            ali_rnn_2n = layers.LayerNormalization(name='layer_norm_rnn2')(ali_rnn_2)
        else:
            ali_rnn_2n = ali_rnn_2
    else:
        ali_rnn_1 = layers.Bidirectional(
            layers.LSTM(model_config.lstm_nodes[0], return_sequences=True, activation='tanh',
                        recurrent_activation='sigmoid', name="sequence_LSTM_1"))(ali_input)
        if model_config.layers_norm:
            ali_rnn_1n = layers.LayerNormalization(name='layer_norm_rnn1')(ali_rnn_1)
        else:
            ali_rnn_1n = ali_rnn_1
        ali_rnn_2 = layers.Bidirectional(
            layers.LSTM(model_config.lstm_nodes[1], return_sequences=True, activation='tanh',
                        recurrent_activation='sigmoid', name="sequence_LSTM_2"))(ali_rnn_1n)
        if model_config.layers_norm:
            ali_rnn_2n = layers.LayerNormalization(name='layer_norm_rnn2')(ali_rnn_2)
        else:
            ali_rnn_2n = ali_rnn_2


    if model_config.n_eigenvec:
        # dense on phylo data
        phy_dnn_1 = layers.Dense(model_config.nn_phylo_features, activation='relu', name="phylo_FC_1")(phy_input)

    #--- block w shared prms
    site_dnn_1 = layers.Dense(model_config.nn_shared_rate_tl, activation='swish', name="site_NN")

    if model_config.separate_block_nn:
        site_dnn_1_tl = layers.Dense(model_config.nn_shared_rate_tl, activation='swish', name="site_NN_tl")


    print("Creating blocks...")
    if model_config.n_eigenvec:
        comb_outputs = [tf.concat((layers.Flatten()(i), phy_dnn_1), 1) for i in tf.split(ali_rnn_2n,
                                                                                         model_config.n_sites, axis=1)]
    else:

        class EmbeddedLayer(Layer):
            def call(self, x):
                return tf.split(x, model_config.n_sites, axis=1)

        comb_outputs = []
        for i in EmbeddedLayer()(ali_rnn_2n):
            x = layers.Flatten()(i)
            comb_outputs.append(x)

        #comb_outputs = [layers.Flatten()(i) for i in tf.split(ali_rnn_2n, model_config.n_sites, axis=1)]

    site_sp_dnn_1_list = [site_dnn_1(i) for i in comb_outputs]
    if model_config.separate_block_nn:
        site_sp_dnn_1_list_tl = [site_dnn_1_tl(i) for i in comb_outputs]

    # add second DNN layer (1 node per site)
    if model_config.pool_per_site:
        site_dnn_2 = layers.Dense(1, activation='swish', name="site_rate_hidden2")
        if model_config.separate_block_nn:
            site_sp_dnn_2_list = [site_dnn_2(i) for i in site_sp_dnn_1_list_tl]
        else:
            site_sp_dnn_2_list = [site_dnn_2(i) for i in site_sp_dnn_1_list]
        concat_1 = layers.concatenate(site_sp_dnn_2_list)
    else:
        concat_1 = layers.concatenate(site_sp_dnn_1_list)
    print("done")
    #---

    # Merge all available features into a single large vector via concatenation
    # concat_1 = layers.concatenate([ali_rnn_3, phy_dnn_1])
    outputs = []
    loss = {}
    loss_w = {}
    # output 1: per-site rate
    if 'per_site_rate' in model_config.output_list:
        site_rate_1 = layers.Dense(model_config.nn_rate[0], activation='swish', name="site_rate_hidden")
        if len(model_config.nn_rate) > 1:
            print("Warning: only single nn_rate layer is currently supported!")
        site_rate_1_list = [site_rate_1(i) for i in site_sp_dnn_1_list]
        rate_pred_nn = layers.Dense(1, activation=model_config.output_f[0], name="per_site_rate_split")
        rate_pred_list = [rate_pred_nn(i) for i in site_rate_1_list]
        # rate_pred =  layers.Dense(1, activation='linear', name="per_site_rate")(layers.concatenate(rate_pred_list))

        if not model_config.mean_normalize_rates:
            rate_pred = layers.Flatten(name="per_site_rate")(layers.concatenate(rate_pred_list))
        else:
            def mean_rescale(x):
                return x / tf.reduce_mean(x, axis=1, keepdims=True)

            generic_utils.get_custom_objects().update({'mean_rescale': Activation(mean_rescale)})
            rate_pred_tmp = layers.Flatten(name="per_site_rate_tmp")(layers.concatenate(rate_pred_list))
            rate_pred = layers.Activation(mean_rescale, name='per_site_rate')(rate_pred_tmp)

        outputs.append(rate_pred)
        loss['per_site_rate'] = keras.losses.MeanSquaredError()
        loss_w["per_site_rate"] = model_config.loss_weights[0]

    # output 2: model test (e.g. JC, HKY, GTR)
    if 'sub_model' in model_config.output_list:
        subst_model_1 = layers.Dense(model_config.nn_sub_model[0], activation='relu', name="sub_model_hidden")(concat_1)
        sub_model_pred = layers.Dense(len(model_config.sub_models), activation=model_config.output_f[1], name="sub_model")(subst_model_1)
        outputs.append(sub_model_pred)
        loss['sub_model'] = keras.losses.CategoricalCrossentropy(from_logits=False)
        loss_w['sub_model'] = model_config.loss_weights[1]

    # output 3: tree length
    if 'tree_len' in model_config.output_list:
        tree_len_1 = layers.Dense(model_config.nn_tl[0], activation='swish', name="tree_len_hidden")(concat_1)
        if len(model_config.nn_tl) == 1:
            tree_len_pred = layers.Dense(1, activation=model_config.output_f[2], name="tree_len")(tree_len_1)
        else:
            if model_config.nn_tl[1] > 0:
                tree_len_2 = layers.Dense(model_config.nn_tl[1], activation='swish', name="tree_len_hidden_2")(tree_len_1)
                tree_len_pred = layers.Dense(1, activation=model_config.output_f[2], name="tree_len")(tree_len_2)
            else:
                tree_len_pred = layers.Dense(1, activation=model_config.output_f[2], name="tree_len")(tree_len_1)
        outputs.append(tree_len_pred)
        loss['tree_len'] = keras.losses.MeanSquaredError()
        loss_w['tree_len'] = model_config.loss_weights[2]

    # output combined
    if 'per_site_abs_rate' in model_config.output_list:
        absolute_rate = layers.Multiply(name='per_site_abs_rate')([rate_pred, tree_len_pred])
        outputs.append(absolute_rate)
        loss['per_site_abs_rate'] = keras.losses.MeanSquaredError()
        loss_w['per_site_abs_rate'] = 1


    # Instantiate an end-to-end model predicting both rates and substitution model
    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        loss_weights=loss_w
    )

    if print_summary:
        print(model.summary())

    print("N. model parameters:", model.count_params())

    return model



def build_rnn_multiple_output_prm_share(n_sites, n_species,
                                        n_eigenvec,
                                        n_onehot=4,
                                        sub_models=None,
                                        loss_weights=None,
                                        nodes = None,
                                        optimizer=keras.optimizers.RMSprop(1e-3)
                              ):
    if nodes is None:
        nodes = [64, 8, 8, 12, 32, 64]
    if sub_models is None:
        sub_models = [0, 1, 2] # ['JC', 'HKY', 'GTR']
    if loss_weights is None:
        loss_weights = [1., 1.]
    ali_input = keras.Input(shape=(n_sites, n_species * n_onehot,), name="sequence_data")
    phy_input = keras.Input(shape=(n_species * n_eigenvec,), name="eigen_vectors")

    # lstm on sequence data
    ali_rnn_1 = layers.LSTM(nodes[0], return_sequences=True, activation='tanh',
                            recurrent_activation='sigmoid', name="sequence_LSTM_1")(ali_input)
    ali_rnn_2 = layers.LSTM(nodes[1], return_sequences=True, activation='tanh',
                            recurrent_activation='sigmoid', name="sequence_LSTM_2")(ali_rnn_1)

    # dense on phylo data
    phy_dnn_1 = layers.Dense(nodes[2], activation='relu', name="phylo_FC_1")(phy_input)

    #--- block w shared prms
    site_dnn_1 = layers.Dense(nodes[3], activation='relu', name="site_NN")

    print("Creating blocks...")
    comb_outputs = [tf.concat((layers.Flatten()(i), phy_dnn_1), 1) for i in tf.split(ali_rnn_2, n_sites, axis=1)]

    site_sp_dnn_1_list = [site_dnn_1(i) for i in comb_outputs]

    concat_1 = layers.concatenate(site_sp_dnn_1_list)
    print("done")
    #---

    # Merge all available features into a single large vector via concatenation
    # concat_1 = layers.concatenate([ali_rnn_3, phy_dnn_1])

    # output 1: per-site rate
    site_rate_1 = layers.Dense(nodes[4], activation='relu', name="site_rate_hidden")
    site_rate_1_list = [site_rate_1(i) for i in site_sp_dnn_1_list]
    rate_pred_nn = layers.Dense(1, activation='linear', name="per_site_rate_split")
    rate_pred_list = [rate_pred_nn(i) for i in site_rate_1_list]
    # rate_pred =  layers.Dense(1, activation='linear', name="per_site_rate")(layers.concatenate(rate_pred_list))
    rate_pred =  layers.Flatten(name="per_site_rate")(layers.concatenate(rate_pred_list))

    # output 2: model test (e.g. JC, HKY, GTR)
    subst_model_1 = layers.Dense(nodes[5], activation='relu', name="sub_model_hidden")(concat_1)
    sub_model_pred = layers.Dense(len(sub_models), activation='softmax', name="sub_model")(subst_model_1)

    # Instantiate an end-to-end model predicting both rates and substitution model
    model = keras.Model(
        inputs=[ali_input, phy_input],
        outputs=[rate_pred, sub_model_pred],
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "per_site_rate": keras.losses.MeanSquaredError(),
            "sub_model": keras.losses.CategoricalCrossentropy(from_logits=False),
        },
        loss_weights={"per_site_rate": loss_weights[0], "sub_model": loss_weights[1]}
    )

    return model


def build_rnn_multiple_output(n_sites, n_species,
                              n_eigenvec,
                              n_onehot=4,
                              sub_models=None,
                              loss_weights=None
                              ):
    if sub_models is None:
        sub_models = [0, 1, 2] # ['JC', 'HKY', 'GTR']
    if loss_weights is None:
        loss_weights = [1., 1.]
    ali_input = keras.Input(shape=(n_sites, n_species * n_onehot,), name="sequence_data")
    phy_input = keras.Input(shape=(n_species * n_eigenvec,), name="eigen_vectors")

    # lstm on sequence data
    ali_rnn_1 = layers.LSTM(64, return_sequences=True, activation='tanh',
                            recurrent_activation='sigmoid', name="sequence_LSTM_1")(ali_input)
    ali_rnn_2 = layers.LSTM(1, return_sequences=True, activation='tanh',
                            recurrent_activation='sigmoid', name="sequence_LSTM_2")(ali_rnn_1)
    ali_rnn_3 = layers.Flatten()(ali_rnn_2)

    # dense on phylo data
    phy_dnn_1 = layers.Dense(64, activation='relu', name="phylo_FC_1")(phy_input)

    # Merge all available features into a single large vector via concatenation
    concat_1 = layers.concatenate([ali_rnn_3, phy_dnn_1])

    # output 1: per-site rate
    site_rate_1 = layers.Dense(n_sites, activation='relu', name="site_rate_hidden")(concat_1)
    rate_pred = layers.Dense(n_sites, activation='linear', name="per_site_rate")(site_rate_1)

    # output 2: model test (e.g. JC, HKY, GTR)
    subst_model_1 = layers.Dense(64, activation='relu', name="sub_model_hidden")(concat_1)
    sub_model_pred = layers.Dense(len(sub_models), activation='softmax', name="sub_model")(subst_model_1)

    # Instantiate an end-to-end model predicting both rates and substitution model
    model = keras.Model(
        inputs=[ali_input, phy_input],
        outputs=[rate_pred, sub_model_pred],
    )

    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss={
            "per_site_rate": keras.losses.MeanSquaredError(),
            "sub_model": keras.losses.CategoricalCrossentropy(from_logits=False),
        },
        loss_weights={"per_site_rate": loss_weights[0], "sub_model": loss_weights[1]}
    )
    return model


def build_rnn(Xt,
              lstm_nodes=None,
              dense_nodes=None,
              dense_act_f='relu',
              output_nodes=1,
              output_act_f='softplus',
              loss_f='mse',
              verbose=1,
              model=None,
              return_sequences=True,
              learning_rate=0.001):
    if lstm_nodes is None:
        lstm_nodes = [64, 32]
    if dense_nodes is None:
        dense_nodes = [32]
    if model is None:
        model = keras.Sequential()

    model.add(
        layers.Bidirectional(layers.LSTM(lstm_nodes[0],
                                         return_sequences=return_sequences,
                                         activation='tanh',
                                         recurrent_activation='sigmoid'),
                             input_shape=Xt.shape[1:])
    )
    for i in range(1, len(lstm_nodes)):
        model.add(layers.Bidirectional(layers.LSTM(lstm_nodes[i],
                                                   return_sequences=return_sequences,
                                                   activation='tanh',
                                                   recurrent_activation='sigmoid')))
    for i in range(len(dense_nodes)):
        model.add(layers.Dense(dense_nodes[i],
                               activation=dense_act_f))

    model.add(layers.Dense(output_nodes,
                           activation=output_act_f))
    if verbose:
        print(model.summary())

    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss_f,
                  optimizer=opt,
                  metrics=['mae', 'mse', 'msle', 'mape'])
    return model


def build_rnn_one(Xt,
                  lstm_nodes=None,
                  dense_nodes=None,
                  dense_act_f='relu',
                  output_nodes=1,
                  output_act_f='softplus',
                  loss_f='mse',
                  verbose=1,
                  learning_rate=0.001):
    if lstm_nodes is None:
        lstm_nodes = [64, 32]
    if dense_nodes is None:
        dense_nodes = [32]
    model = keras.Sequential()

    model.add(
        layers.Bidirectional(layers.LSTM(lstm_nodes[0],
                                         return_sequences=False,
                                         activation='tanh',
                                         recurrent_activation='sigmoid'),
                             input_shape=Xt.shape[1:])
    )
    # for i in range(1, len(lstm_nodes)):
    #     model.add(layers.Bidirectional(layers.LSTM(lstm_nodes[i],
    #                                                return_sequences=True,
    #                                                activation='tanh',
    #                                                recurrent_activation='sigmoid')))
    for i in range(len(dense_nodes)):
        model.add(layers.Dense(dense_nodes[i],
                               activation=dense_act_f))

    model.add(layers.Dense(output_nodes,
                           activation=output_act_f))
    if verbose:
        print(model.summary())

    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss_f,
                  optimizer=opt,
                  metrics=['mae', 'mse'],
                  )
    return model

def fit_rnn(Xt, Yt, model,
            criterion="val_loss",
            patience=10,
            verbose=1,
            batch_size=100,
            max_epochs=1000,
            validation_split=0.2
            ):
    early_stop = keras.callbacks.EarlyStopping(monitor=criterion,
                                               patience=patience,
                                               restore_best_weights=True)
    history = model.fit(Xt, Yt,
                        epochs=max_epochs,
                        validation_split=validation_split,
                        verbose=verbose,
                        callbacks=[early_stop],
                        batch_size=batch_size
                        )
    return history


def save_rnn_model(wd, history, model, feature_rescaler=None, filename=""):
    # save rescaler
    if feature_rescaler is not None:
        with open(os.path.join(wd, "rnn_rescaler" + filename + ".pkl"), 'wb') as output:  # Overwrites any existing file.
            pkl.dump(feature_rescaler(1), output, pkl.HIGHEST_PROTOCOL)
    # save training history
    with open(os.path.join(wd, filename + "_history" + ".pkl"), 'wb') as output:  # Overwrites any existing file.
        pkl.dump(history.history, output, pkl.HIGHEST_PROTOCOL)
    # save model
    tf.keras.models.save_model(model, os.path.join(wd, filename + '_model.keras'))


def load_rnn_model(wd, filename=""):
    model = tf.keras.models.load_model(os.path.join(wd, filename))
    return model


def get_r2(x, y):
    _, _, r_value, _, _ = scipy.stats.linregress(x, y)
    return r_value**2


def get_mse(x, y):
    mse = np.mean((x - y)**2)
    return mse


def get_avg_r2(Ytrue, Ypred):
    r2 = []
    if len(Ypred.shape) == 3:
        Ypred = Ypred[:, :, 0]

    for i in range(Ytrue.shape[0]):
        x = Ytrue[i]
        y = Ypred[i, :]
        r2.append(get_r2(x[x > 0], y[x > 0]))
    res = {'mean r2': np.nanmean(r2),   # change to nanmean? check if this works for sqs data.
           'min r2': np.nanmin(r2),
           'max r2': np.nanmax(r2),
           'std r2': np.nanstd(r2)}
    return res


def get_avg_mse(Ytrue, Ypred):
    mse = []
    if len(Ypred.shape) == 3:
        Ypred = Ypred[:, :, 0]

    for i in range(Ytrue.shape[0]):
        x = Ytrue[i]
        y = Ypred[i, :]
        mse.append(get_mse(x[x > 0], y[x > 0]))
    res = {'mean mse': np.nanmean(mse),
           'min mse': np.nanmin(mse),
           'max mse': np.nanmax(mse),
           'std mse': np.nanstd(mse)}
    return res

if __name__ == "__main__":
    # the input data
    n_instances = 500
    img_dimension = [40, 40]
    channels = 3
    n_additional_features = 5
    # this is our image data
    x1_train = np.random.uniform(size=[n_instances] + img_dimension + [channels])
    # these are additional 'manually' generated features
    x2_train = np.random.uniform(size=[n_instances, n_additional_features])
    labels = np.random.uniform(size=n_instances)


    # rnn layers
    n_instances = 500
    n_sites = 100
    n_species = 20
    n_onehot = 4
    n_eigenvec = 3
    """
    input
    
    
    """

    x1_train = np.random.uniform(size=(n_instances, n_sites, n_species * (n_onehot + n_eigenvec)))
    x2_train = np.random.uniform(size=(n_instances, n_species, n_eigenvec))

    lstm_nodes = [20,5]
    dense_nodes = [10]
    dense_act_f = 'relu'
    output_act_f = 'softplus'
    architecture_rnn = []
    rnn_output_shape = n_species

    architecture_rnn.append(
        layers.Bidirectional(layers.LSTM(lstm_nodes[0],
                                         return_sequences=True,
                                         activation='tanh',
                                         recurrent_activation='sigmoid'),
                             input_shape=x1_train.shape[1:])
    )
    for i in range(1, len(lstm_nodes)):
        architecture_rnn.append(layers.Bidirectional(layers.LSTM(lstm_nodes[i],
                                                                 return_sequences=True,
                                                                 activation='tanh',
                                                                 recurrent_activation='sigmoid')))
    for i in range(len(dense_nodes)):
        architecture_rnn.append(layers.Dense(dense_nodes[i],
                               activation=dense_act_f))

    architecture_rnn.append(layers.Dense(rnn_output_shape,
                           activation=output_act_f))

    rnn_model = tf.keras.Sequential(architecture_rnn)

    # fully connected NN
    dense_nodes_fc = [4, 1]

    architecture_fc = []
    architecture_fc.append(tf.keras.layers.Dense(dense_nodes_fc[0],
                                                 activation='relu',
                                                 input_shape=x2_train.shape[1:]))

    architecture_fc.append(layers.Dense(dense_nodes_fc[1],
                                        activation='tanh'))



    fc_model = tf.keras.Sequential(architecture_fc)

    rnn_output = rnn_model(x1_train) # shape = (n_instances, n_sites, n_species)
    fc_output = fc_model(x2_train) # shape (transposed as below) = (n_instances, n_output_nodes, n_species)

    concatenatedFeatures = tf.keras.layers.Concatenate(axis=2)([rnn_output, tf.transpose(fc_output, perm=[0,2,1])])

    concat_model_architecture = []
    concat_model_architecture.append(tf.keras.layers.Dense(dense_nodes_fc[0],
                                                 activation='relu',
                                                 input_shape=x2_train.shape[1:]))

    # output = fc_model(cnn_output)
    output = fc_model(concatenatedFeatures)

    model = tf.keras.models.Model([input1, input2], output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
    model.summary()


    #--- CNN + Dense
    # the input data
    n_instances = 500
    img_dimension = [40, 40]
    channels = 3
    n_additional_features = 5
    # this is our image data
    x1_train = np.random.uniform(size=[n_instances] + img_dimension + [channels])
    # these are additional 'manually' generated features
    x2_train = np.random.uniform(size=[n_instances, n_additional_features])
    labels = np.random.uniform(size=n_instances)

    #
    # ___________________________________BUILD THE CNN MODEL____________________________________
    # convolution layers (feature generation)
    architecture_conv = []
    architecture_conv.append(tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), activation='relu', padding='valid'))
    architecture_conv.append(tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same'))
    architecture_conv.append(tf.keras.layers.Flatten())
    conv_model = tf.keras.Sequential(architecture_conv)

    # fully connected NN
    architecture_fc = []
    architecture_fc.append(tf.keras.layers.Dense(40, activation='relu'))
    architecture_fc.append(tf.keras.layers.Dense(20, activation='relu'))
    architecture_fc.append(tf.keras.layers.Dense(1, activation='softplus'))  # sigmoid or tanh or softplus
    fc_model = tf.keras.Sequential(architecture_fc)

    # define the input layer and apply the convolution part of the NN to it
    input1 = tf.keras.layers.Input(shape=x1_train.shape[1:])
    cnn_output = conv_model(input1)

    # define the second input that will come in after the convolution
    input2 = tf.keras.layers.Input(shape=(n_additional_features,))
    concatenatedFeatures = tf.keras.layers.Concatenate(axis=1)([cnn_output, input2])

    # output = fc_model(cnn_output)
    output = fc_model(concatenatedFeatures)

    model = tf.keras.models.Model([input1, input2], output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
    model.summary()

def train_on_sql_batch(model, epochs,batch_size, sqlite_fn, patience=5, early_stopping = False, verbose=True):

    history = []  # Store loss and accuracy per epoch
    best_val_loss = float("inf")
    wait = 0  # Counter for early stopping
    patience = patience

    for epoch in range(epochs):

        epoch_loss, num_batches = 0, 0

        batch_gen = sqlite_data_generator(sqlite_fn, batch_size)

        for X_batch, y_batch in batch_gen:
            t = model.train_on_batch(X_batch, y_batch)  # Train on a single batch
            epoch_loss += t[0]
            num_batches += 1

        epoch_loss = epoch_loss / num_batches
        history.append(epoch_loss)

        if verbose:
            print("Epoch {} - Loss {}".format(epoch,epoch_loss))

        # Early stopping check
        if history[-1] < best_val_loss:
            best_val_loss = history[-1]
            wait = 0  # Reset patience counter
        else:
            wait += 1  # Increment patience counter
            if wait >= patience and early_stopping:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break  # Stop training

    return  model, history

