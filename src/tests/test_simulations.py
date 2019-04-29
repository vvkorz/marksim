
import unittest
from src.marksim.simulate import Simulator
from src.marksim.tpm import TPM
import numpy as np


class TestSIM(unittest.TestCase):
    """
    Tests simulations
    """

    def test_simulate(self,):
        """
        .. note:
           simulation tests are not developed.

        """
        array = np.array([[4., 2., 2., 0., 0., 0., np.nan, np.nan, 4., 3.],
                           [np.nan, 2., np.nan, 3., np.nan, 2., 1., 3., 3., 4.],
                           [0., 0., 3., np.nan, 3., 3., np.nan, 2., 2., 0.],
                           [1., np.nan, 0., np.nan, 1., np.nan, 0., 2., 0., 0.],
                           [2., 1., 2., 0., 0., 1., 3., 4., np.nan, 1.],
                           [3., 1., 3., 2., 2., 1., 0., 0., np.nan, np.nan],
                           [np.nan, 4., 4., 4., 4., 0., 2., 0., 1., np.nan],
                           [2., 0., 1., 1., np.nan, 2., 4., 1., 0., 2.],
                           [1., np.nan, 1., 2., 2., np.nan, 1., 3., 2., 2.],
                           [0., 3., 0., 1., 1., 4., 2., 1., 1., 1.]])
        n_sim = 2
        n_states = 4
        mask = np.ma.masked_invalid(array).mask
        tpm_matrix = TPM.calculate_tpm(array, n_states=n_states)
        sim = Simulator(tpm_matrix=tpm_matrix, mask=mask, n_sim=n_sim)
        sim_array = sim.simulate()
        self.assertEqual(sim_array.shape, (2, 10, 10))
