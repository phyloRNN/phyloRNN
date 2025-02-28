# Import the memory profiler
from memory_profiler import profile

import phyloRNN as pn

@profile
def train():

    # setup model architecture
    model_config = pn.rnn_config(n_sites=200, n_taxa=20)  # default settings

    # build model
    model = pn.build_rnn_model(model_config, optimizer=pn.keras.optimizers.RMSprop(1e-3), print_summary=False)

    model_trained, history =  pn.train_on_sql_batch(model, 20, 100, "../data/training_data_taxa_20_sites200_2000.db")

    print(history)

'''
To run:
    mprof run filename.py
    mprof plot 
or 
    python -m memory_profiler normal_training.py

'''
if __name__ == '__main__':
    train()