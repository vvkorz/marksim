
import unittest
from src.marksim.simulate import Simulator
from src.marksim.tpm import TPM
from src.marksim.analyse import SimulationAnalysis
import numpy as np


class TestAnalytics(unittest.TestCase):
    """
    Tests analysis of simulations
    """

    def test_analyse_all_simulations(self,):
        """
        testing a simple set up here
        """
        # here the original array contains states in which instances are observed

        original = np.array([[np.nan,  1.,  0.,  1.,  2.,  1.,  2.,  2.,  2., np.nan],
                             [0., np.nan,  2., np.nan, np.nan, 0.,  4.,  2., np.nan,  0.],
                             [1.,  0.,  4.,  2.,  3.,  1. , 2.,  0., np.nan, np.nan],
                             [np.nan,  3., np.nan,  4.,  0.,  2.,  0., np.nan,  0.,  1.],
                             [4.,  1.,  1.,  0.,  1., np.nan,  1.,  4.,  3.,  1.],
                             [1.,  2.,  3.,  1.,  2.,  4.,  0.,  3.,  4.,  2.],
                             [3.,  0.,  1.,  3.,  0.,  2.,  3., np.nan,  0.,  3.],
                             [2.,  2.,  2., np.nan, np.nan, np.nan, np.nan,  1.,  1.,  4.],
                             [0.,  4., np.nan,  2.,  1.,  3.,  1.,  0.,  2.,  2.],
                             [2., np.nan,  0.,  0.,  4.,  0., np.nan,  1.,  1.,  0.]])
        simulated = np.array([[[np.nan,  2.,  2.,  1.,  3.,  1.,  1.,  2.,  3., np.nan],
                              [2., np.nan,  0., np.nan, np.nan,  1.,  0.,  2., np.nan,  0.],
                              [2.,  0.,  2.,  4.,  2.,  2.,  2.,  3., np.nan, np.nan],
                              [np.nan,  1., np.nan,  2.,  2.,  2.,  1., np.nan,  4.,  0.],
                              [3.,  1.,  1., 0.,  2., np.nan,  4.,  0.,  1.,  2.,],
                              [4.,  0.,  3.,  1.,  2.,  0.,  2.,  4.,  0.,  1.],
                              [4.,  0.,  4.,  1.,  1.,  0.,  3., np.nan,  2.,  1.],
                              [2.,  2.,  2., np.nan, np.nan, np.nan, np.nan,  0.,  2.,  2.],
                              [3.,  0., np.nan,  0.,  0.,  0.,  4.,  0.,  1.,  2.],
                              [0., np.nan,  0.,  1.,  4.,  2., np.nan,  2.,  1.,  3.]],
                             [[np.nan,  1.,  0.,  4.,  2.,  2.,  3.,  0.,  2., np.nan],
                              [2., np.nan,  0., np.nan, np.nan,  1.,  2.,  1., np.nan,  1.],
                              [2.,  2.,  2.,  2.,  0.,  1.,  1.,  0., np.nan, np.nan],
                              [np.nan,  2., np.nan,  4.,  0.,  2.,  0., np.nan,  2.,  3.],
                              [1.,  1.,  2.,  3.,  1., np.nan,  4.,  0.,  4.,  3.],
                              [1.,  4.,  0.,  4.,  3.,  0.,  4.,  3.,  1.,  1.],
                              [1.,  2.,  1.,  2.,  3.,  1.,  2., np.nan,  2.,  0.],
                              [4.,  0.,  1., np.nan, np.nan, np.nan, np.nan,  4.,  0.,  4.],
                              [1.,  2., np.nan,  2.,  2.,  3.,  4.,  1.,  2.,  3.],
                              [2., np.nan,  3.,  1.,  0.,  1., np.nan,  2.,  2.,  2.]],
                             [[np.nan,  4.,  3.,  1.,  2.,  2.,  1.,  0.,  4., np.nan],
                              [1., np.nan,  1., np.nan, np.nan,  1.,  2.,  2., np.nan,  0.],
                              [1.,  2.,  1.,  0.,  1.,  3.,  4.,  1., np.nan, np.nan],
                              [np.nan,  0., np.nan,  1.,  0.,  2.,  3., np.nan,  2.,  0.],
                              [2.,  4., 1.,  2.,  2., np.nan,  0.,  1.,  2.,  2.],
                              [1.,  3.,  1.,  0.,  1.,  0.,  2.,  2.,  3.,  0.],
                              [1.,  2.,  0.,  4.,  0.,  2.,  0., np.nan,  1.,  0.],
                              [4.,  0.,  3., np.nan, np.nan, np.nan, np.nan,  0.,  4.,  0.],
                              [4.,  0., np.nan,  2.,  1.,  3.,  0.,  1.,  2.,  0.],
                              [2., np.nan,  0.,  4.,  2.,  3., np.nan,  4.,  0.,  1.]]])

        a = SimulationAnalysis(original, top=4, p=10)
        result = a.analyse_all_simulations(simulated)

        correct_result = {8: {'more': 0, 'equal': 2, 'less': 1}, 3: {'more': 0, 'equal': 1, 'less': 2},
                          4: {'more': 0, 'equal': 0, 'less': 3}, 2: {'more': 0, 'equal': 2, 'less': 1},
                          5: {'more': 0, 'equal': 1, 'less': 2}, 10: {'more': 1, 'equal': 0, 'less': 2},
                          7: {'more': 3, 'equal': 0, 'less': 0}}
        self.assertDictEqual(result, correct_result)

