import os
import phyloRNN as pn


if __name__ == '__main__':

    cpu = 1
    num_site = 1000
    num_taxa = 500
    datapoint = [40,80,200,400,1000]

    for size in datapoint :

        print('Simulation {} alignment'.format(size))

        sim = pn.simulator(
            n_taxa=num_taxa,
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
                       n_sims = size,
                       data_name = "training_data_taxa{}_site{}_sim{}_fake".format(num_taxa,num_site,size),
                       base_seed = 1234)

        pn.simulate_parallel(sim, add_day_tag=False)

        print('Simulation saved')
