import os
import phyloRNN as pn
import time
from numba import cuda


if __name__ == '__main__':

    time_point = []
    cpu = 10
    num_sims_per_cpu = 100
    num_site = 1000
    datapoint = [10,50,100,500,1000,10000]
    epoch= 20
    batch_size = 50

    for size in datapoint:

        print('Training {} taxa'.format(size))
        start = time.process_time()

        training_file = os.path.join(os.getcwd(), "training_data_taxa{}_site{}_sim{}_fake.db".format(size,num_site,num_sims_per_cpu*cpu))

        # setup model architecture
        model_config = pn.rnn_config(n_sites=num_site, n_taxa=size) # default settings

        # build model
        model = pn.build_rnn_model(model_config,
                                   optimizer=pn.keras.optimizers.RMSprop(1e-3),
                                   print_summary=False)

        print('Model built.')


        model_trained, history = pn.train_on_sql_batch(model, epoch, batch_size, training_file, patience=100)
        end = time.process_time()
        time_point.append(end-start)

        print('Model trained in {} Sec'.format(end-start))

    print(time_point, datapoint)

    device = cuda.get_current_device()
    device.reset()

