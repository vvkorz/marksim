
import unittest
from src.marksim.TPM import tpm
import numpy as np


class TestTPM(unittest.TestCase):
    """
    Tests for the transition probability matrix calculation
    """

    def nan_equal(self, a, b):
        """
        compare two numpy arrays with NaN

        :param a: np.array
        :param b: np.array
        :return: Boolean
        """
        try:
            np.testing.assert_equal(a, b)
        except AssertionError:
            return False
        return True

    def test_functionality(self):
        test_data = np.random.rand(250, 20)
        test_data[2, 0] = None
        a = tpm.TPM()
        b = a.parse(test_data)
        pass

    def test_invalid_array_shape(self):
        """
        should only work with 2D
        """
        a = tpm.TPM()
        self.assertRaises(AssertionError, a.parse, np.random.rand(250, 20, 80))

    def test_invalid_array_states(self):
        """
        should only work with 2D arrays where number of rows is bigger that number of TPM states
        """
        a = tpm.TPM()
        a.n_states = 100
        self.assertRaises(AssertionError, a.parse, np.random.rand(60, 20))

    def test_get_bins(self):
        benchmarkarray = np.array([5, 7, 0, 0, 6, 2, 1, 8, 7, 4, 3, 5, 6, 4, 9, 2, 8, 3, 1, 9])
        array = np.array([10, 14, 0, 1, 12, 4, 3, 17, 15, 9, 7, 11, 13, 8, 18, 5, 16, 6, 2, 19])
        self.assertTrue(np.array_equal(benchmarkarray, list(tpm.TPM.get_bins(array, bins=10))))

    def test_calculate_states(self):
        array = np.array([np.nan,  1,  2, np.nan, np.nan,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, np.nan, 17, 18, 19])
        benchmarkarray = np.array([np.nan,  0.,  0., np.nan, np.nan,  1.,  1.,  2.,  2.,  3.,  3.,  4.,  4.,  5.,  5.,  6., np.nan,  7.,  8.,  9.])
        self.assertTrue(self.nan_equal(benchmarkarray, tpm.TPM.calculate_states(array, n_states=10)))

    def test_convert_to_states(self):
        array = np.array([[9.,  6.,  6.,  2.,  2.,  1., np.nan, np.nan,  9.,  7.],
                          [np.nan,  7., np.nan,  8., np.nan,  4.,  5.,  8.,  8.,  9.],
                          [0., 1., 7., np.nan, 8., 7., np.nan, 6., 4., 0.],
                          [3., np.nan,  0., np.nan,  4., np.nan,  2.,  4.,  0.,  1.],
                          [7., 4., 4., 1., 0., 3., 8., 9., np.nan, 2.],
                          [8.,  3.,  8.,  7.,  6.,  2.,  0.,  0., np.nan, np.nan],
                          [np.nan, 9., 9., 9., 9., 0., 6., 1., 2., np.nan],
                          [5.,  2.,  3.,  5., np.nan,  5., 9.,  3.,  1.,  5.],
                          [4., np.nan, 2., 6., 7., np.nan, 3., 7., 6., 4.],
                          [1.,  8.,  1.,  3.,  3.,  9.,  7.,  2.,  3.,  3.]])
        benchmarkarray = np.array([[ 4.,  2.,  2.,  0.,  0.,  0., np.nan, np.nan,  4.,  3.],
                          [np.nan,  2., np.nan,  3., np.nan,  2.,  1.,  3.,  3.,  4.],
                          [ 0.,  0.,  3., np.nan,  3.,  3., np.nan,  2.,  2.,  0.],
                          [ 1., np.nan,  0., np.nan,  1., np.nan,  0.,  2.,  0.,  0.],
                          [ 2.,  1.,  2.,  0.,  0.,  1.,  3.,  4., np.nan,  1.],
                          [ 3.,  1.,  3.,  2.,  2.,  1.,  0.,  0., np.nan, np.nan],
                          [np.nan,  4.,  4.,  4.,  4.,  0.,  2.,  0.,  1., np.nan],
                          [ 2.,  0.,  1.,  1., np.nan,  2.,  4.,  1.,  0.,  2.],
                          [ 1., np.nan,  1.,  2.,  2., np.nan,  1.,  3.,  2.,  2.],
                          [ 0.,  3.,  0.,  1.,  1.,  4.,  2.,  1.,  1., 1.]])
        self.assertTrue(self.nan_equal(benchmarkarray, tpm.TPM.convert_to_states(array, n_states=5)))
