import os
import phyloRNN as pn


if __name__ == '__main__':

    cpu = 10
    num_sims_per_cpu = 100
    num_site = 1000
    datapoint = [10,50,100,500,1000,10000,20000,50000]

    for size in datapoint :

        print('Simulation {} taxa'.format(size))

        sim = pn.simulator(
            n_taxa=size,
            n_sites=num_site,
            n_eigen_features=0,
            min_rate=0,  #
            freq_uncorrelated_sites=1,
            freq_mixed_models=0,
            freq_seqgen_codon=1,
            store_mixed_model_info=True,
            tree_builder='nj',  # 'upgma'
            subs_model_per_block=False,  # if false same subs model for all blocks
            phyml_path=None,  # path to phyml and seq binaries
            seqgen_path=None, # if None, it will try to use
            ali_path=None,  # system-wide installed software
            DEBUG=False,
            verbose=False,
            format_output='sqlite',  # 'sqlite' , 'npz, or 'both'
            fake=True
        )

        print('Simulator build')

        sim.reset_prms(CPUs = cpu,
                       n_sims = num_sims_per_cpu,
                       data_name = "training_data_taxa{}_site{}_sim{}_fake".format(size,num_site,cpu*num_sims_per_cpu),
                       base_seed = 1234)
        pn.simulate_parallel(sim, add_day_tag=False)

        print('Simulation saved')
