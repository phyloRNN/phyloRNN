# Import the memory profiler
from memory_profiler import profile

import phyloRNN as pn
import sqlite3
import zlib
import numpy as np
import json




def sqlite_data_generator(db_path, batch_size):

    batch =  None
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

        def _decompdecode(e, dtype, shape=False):
            if shape:
                return np.frombuffer(zlib.decompress(e), dtype=dtype).reshape(shape)
            else:
                return np.frombuffer(zlib.decompress(e), dtype=dtype)

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

        sim, dict_inputs, dict_outputs = pn.rnn_in_out_dictionaries_from_sim(sim=sim, log_rates=False,
                                                                             log_tree_len=True,
                                                                             output_list=['per_site_rate', 'tree_len'],
                                                                             include_tree_features=False, sqlite=True)

        yield dict_inputs, dict_outputs

    conn.close()

@profile
def train():
    batch_size = 500
    epochs = 20

    # setup model architecture
    model_config = pn.rnn_config(n_sites=1000, n_taxa=50)  # default settings

    # build model
    model = pn.build_rnn_model(model_config, optimizer=pn.keras.optimizers.RMSprop(1e-3), print_summary=False)

    # training
    early_stop = pn.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = []  # Store loss and accuracy per epoch
    best_val_loss = float("inf")
    wait = 0  # Counter for early stopping
    patience = 5

    for epoch in range(epochs):

        epoch_loss, num_batches = 0, 0

        batch_gen = sqlite_data_generator("training_data_taxa_50_sites1000_20000.db", batch_size)

        for X_batch, y_batch in batch_gen:
            t = model.train_on_batch(X_batch, y_batch)  # Train on a single batch
            epoch_loss += t[0]
            num_batches += 1

        epoch_loss = epoch_loss / num_batches

        print(epoch_loss , "-" * 35)

        history.append(epoch_loss)

        # Early stopping check
        if history[-1] < best_val_loss:
            best_val_loss = history[-1]
            wait = 0  # Reset patience counter
        else:
            wait += 1  # Increment patience counter
            if wait >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break  # Stop training

'''
To run:
    mprof run filename.py
    mprof plot 
or 
    python -m memory_profiler normal_training.py

'''
if __name__ == '__main__':
    train()