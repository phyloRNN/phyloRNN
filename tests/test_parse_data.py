import unittest
import phyloRNN as pn
import tempfile
import numpy as np
import sqlite3
import zlib


def _decompdecode(e, dtype, shape):
    return np.frombuffer(zlib.decompress(e), dtype=dtype).reshape(shape[1:])


class Simulation(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()


        self.temp_file_path =   self.temp_dir.name + '/temp_data_file'

        self.sim = pn.simulator(
            n_taxa=10,
            n_sites=20,
            n_eigen_features=3,
            min_rate=0,  #
            freq_uncorrelated_sites=0.5,
            freq_mixed_models=0.05,
            store_mixed_model_info=True,
            tree_builder='nj',  # 'upgma'
            subs_model_per_block=False,  # if false same subs model for all blocks
            phyml_path=None,  # path to phyml and seq binaries
            seqgen_path=None,  # if None, it will try to use
            ali_path=None,  # system-wide installed software
            DEBUG=False,
            verbose=False,
            CPUs=2,
            n_sims=5,
            data_name=self.temp_file_path,
            base_seed=1234,
            format_output='both')

        pn.simulate_parallel(self.sim, add_day_tag=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_rnn_in_out_dictionaries_from_sim(self):

        sim, dict_inputs, dict_outputs = pn.rnn_in_out_dictionaries_from_sim(
            '{}.npz'.format(self.temp_file_path),
            log_rates=False,
            log_tree_len=True,
            output_list=['per_site_rate', 'tree_len'],
            include_tree_features=False)

        sim2, dict_inputs2, dict_outputs2 = pn.rnn_in_out_dictionaries_from_sim(
            '{}.db'.format(self.temp_file_path),
            log_rates=False,
            log_tree_len=True,
            output_list=['per_site_rate', 'tree_len'],
            include_tree_features=False,
            sqlite=True)

        for e in sim.keys():
            self.assertEqual(np.array(sim[e]).shape, np.array(sim2[e]).shape)


if __name__ == '__main__':
    unittest.main()
