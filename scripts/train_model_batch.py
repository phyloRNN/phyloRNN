import os
import phyloRNN as pn

training_file = os.path.join(os.getcwd(), "training_data.db")
wd = os.path.dirname(training_file)
model_name = "t50_s1000"

# setup model architecture
model_config = pn.rnn_config(n_sites=1000, n_taxa=50) # default settings

# build model
model = pn.build_rnn_model(model_config,
                           optimizer=pn.keras.optimizers.RMSprop(1e-3),
                           print_summary=False)


model_trained, history = pn.train_on_sql_batch(model, 1000, 100, "../data/training_data_taxa_20_sites200_2000.db", patience=5)

# save model
pn.save_rnn_model(wd=wd, history=history, model=model, filename=model_name)

