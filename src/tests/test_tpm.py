
import unittest
from src.marksim.TPM import tpm
import numpy as np


class TestTPM(unittest.TestCase):
    """
    Tests for the transition probability matrix calculation
    """

    def test_functionality(self):
        test_data = np.random.rand(250, 20)
        test_data[2, 0] = None
        a = tpm.TPM()
        b = a.parse(test_data)
        raise ValueError

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

