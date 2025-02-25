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
        self.temp_file_path = self.temp_dir.name + '/temp_data_file'

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

        self.npz = dict(np.load('{}.npz'.format(self.temp_file_path), allow_pickle=True))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_npz_sqlite_data_identical(self):

        sqlite = sqlite3.connect('{}.db'.format(self.temp_file_path)).cursor()

        # Test simulation and array_info tables are present
        sqlite.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = sqlite.fetchall()
        self.assertEqual(tables, [('simulation',), ('array_info',)])


        # Retrieve and test data from array_info table
        sqlite.execute('SELECT name_, dtype, shape FROM array_info')
        rows = sqlite.fetchall()
        array_info_dict = {name_: {'dtype': dtype, 'shape': eval(shape)} for name_, dtype, shape in rows}
        expected_keys = {'labels_smodel', 'labels_rates', 'labels_tl', 'features_ali', 'features_tree'}
        self.assertEqual(set(array_info_dict.keys()), expected_keys)

        # Test the simulation data shape are similar between .npz and .db
        for column in expected_keys:
            sqlite.execute("SELECT {} FROM simulation".format(column))
            datum = [_decompdecode(row[0], array_info_dict[column]['dtype'], array_info_dict[column]['shape']) for row in sqlite.fetchall()]
            self.assertEqual(self.npz[column].shape, np.array(datum).shape)

        sqlite.close()

if __name__ == '__main__':
    unittest.main()
