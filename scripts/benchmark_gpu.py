import phyloRNN as pn
import datetime
import argparse
import tensorflow as tf

def main(training_file):

    epochs_values = [1,2,5] #10,100,500]

    # load data
    sim, dict_inputs, dict_outputs = pn.rnn_in_out_dictionaries_from_sim(training_file,
                                                                         log_rates=False,
                                                                         log_tree_len=True,
                                                                         output_list=['per_site_rate','tree_len'],
                                                                         include_tree_features=False)


    for epoch in epochs_values:

        # setup model architecture
        model_config = pn.rnn_config(n_sites=1000, n_taxa=50) # default settings

        # build model
        model = pn.build_rnn_model(model_config,
                                   optimizer=pn.keras.optimizers.RMSprop(1e-3),
                                   print_summary=False)

        type_proc = "gpu" if  tf.config.list_physical_devices('GPU') else 'cpu'

        log_dir = f"logs/benchmark/{type_proc}/{epoch}"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        model.fit(dict_inputs, dict_outputs,
                            epochs=epoch,
                            validation_split=0.2,
                            verbose=1, # REMOVE FOR BENCH
                            callbacks=[ tensorboard_callback],
                            batch_size=100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark GPU training")
    parser.add_argument("-t", "--training_file",  type=str, help="Path to the training data file")
    args = parser.parse_args()
    main(args.training_file)