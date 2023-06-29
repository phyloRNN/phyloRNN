import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
np.set_printoptions(suppress=True, precision=3)
import os
import pickle as pkl
import scipy.stats


n_instances = 1
n_sites = 1000
n_species = 50
n_onehot = 4
n_eigenvec = 3


x1_train = np.random.uniform(size=(n_instances, n_sites, n_species * n_onehot))
x2_train = np.random.uniform(size=(n_instances, n_species * n_eigenvec))

# model 1 - RNN
lstm_nodes = [128, 1]
dense_nodes = []
dense_act_f = 'relu'
output_act_f = 'softplus'
rnn_output_shape = 1 # 1 output per site

architecture_rnn = []
architecture_rnn.append(
    layers.Bidirectional(layers.LSTM(lstm_nodes[0],
                                     return_sequences=True,
                                     activation='tanh',
                                     recurrent_activation='sigmoid'),
                         input_shape=x1_train.shape[1:])
)
for i in range(1, len(lstm_nodes)):
    architecture_rnn.append(layers.LSTM(lstm_nodes[i],
                                        return_sequences=True,
                                        activation='tanh',
                                        recurrent_activation='sigmoid'))

# --
# architecture_rnn.append(tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), activation='relu', padding='valid'))
# architecture_rnn.append(tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same'))
# --
architecture_rnn.append(layers.Flatten())

# model 2 - fully connected NN (RNN output + eigenvectors)
dense_nodes_fc = [50, 10]
dense_act_f = 'relu'
architecture_fc = []
architecture_fc.append(layers.Dense(dense_nodes_fc[0], activation=dense_act_f))
architecture_fc.append(layers.Dense(n_sites, activation='linear'))
fc_model = tf.keras.Sequential(architecture_fc)


# define the input layer and apply the RNN part to it
input1 = tf.keras.layers.Input(shape=x1_train.shape[1:])
rnn_model = tf.keras.Sequential(architecture_rnn)
rnn_output = rnn_model(x1_train) # shape = (n_instances, n_sites, n_species)

# define the second input that will come in after the RNN
input2 = tf.keras.layers.Input(shape=x2_train.shape[1:])
concatenate_features = tf.keras.layers.Concatenate(axis=1)([rnn_output, input2])

final_output = fc_model(concatenate_features)

model = tf.keras.models.Model([input1, input2], final_output)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
model.summary()










# concatenate networks
dense_nodes_concat = [50]
dense_act_f = 'relu'
rnn_output = rnn_model(x1_train) # shape = (n_instances, n_sites, n_species)
fc_output = fc_model(x2_train) # shape (transposed as below) = (n_instances, n_output_nodes, n_species)

concatenated_outputs = tf.keras.layers.Concatenate(axis=1)([rnn_output, fc_output])

concat_model_architecture = []
concat_model_architecture.append(tf.keras.layers.Dense(dense_nodes_concat[0],
                                                       activation='relu',
                                                       input_shape=concatenated_outputs.shape[1:]))
for i in range(1, len(dense_nodes_concat)):
    concat_model_architecture.append(layers.Dense(dense_nodes_concat[1], activation=dense_act_f))

concat_model_architecture.append(layers.Dense(n_sites, activation='linear'))
concat_model = tf.keras.Sequential(concat_model_architecture)

# output = fc_model(cnn_output)
final_output = concat_model(concatenated_outputs)

input1 = tf.keras.layers.Input(shape=(x1_train.shape[1:],))
input2 = tf.keras.layers.Input(shape=(x2_train.shape[1:],))
model = tf.keras.models.Model([input1, input2], final_output)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
model.summary()