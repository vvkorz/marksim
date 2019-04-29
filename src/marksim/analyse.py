#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Simulation Analysis
-------------------
"""

from marksim.tpm import TPM
from marksim.utils import utils
from marksim.simulate import Simulator
import array as arr
import time
import numpy as np


class SimulationAnalysis:
    """
    Analyses the results of a simulated markov processes
    """

    def __init__(self, original_panel, top=10, p=0.01):
        self.original_panel = original_panel
        """original panel data"""
        self.top = top
        """definition of superiority to consider"""
        self.p = p
        """confidence interval"""
        self.markov_order = 1

    def disassemble_array(self, matrix):
        """
        takes unbalances panel and splits into balanced panels of fixed bin_size.
        For example we have an unbalanced panel like so:

        >>> array
        [[ 0.  3.  4. nan  0.  0.  1.  0.  0. nan nan  4.]
         [ 3.  0.  3.  2. nan  1.  3.  3.  3.  0.  4.  3.]
         [nan nan nan  3.  2. nan  4.  1.  4.  2.  1.  0.]
         [ 4.  4.  0.  4.  1.  2. nan nan  2.  3.  3.  1.]
         [ 1.  1.  1.  1.  3.  4.  0.  4. nan  1.  2. nan]]

        For each row: the method will first split the row in chunks using np.nan as separator.
        It will then create two arrays from the row (e.g. [0. 3. 4.] and [0.  0.  1.  0.  0.] from first row)
        It will ignore np.NaNs and all arrays which length is more than self.markov_order variable (consider [4.] for the first row)
        The result will be a dictionary with arrays of fixed size as values and their length in columns as keys:

        >>> a = SimulationAnalysis(array)
        >>> result = a.disassemble_array(array)
        >>> for panel_size, panel in result.items():
        ...     print(panel_size)
        ...     print(panel)
        3
        [[0. 3. 4.]]
        5
        [[0. 0. 1. 0. 0.]]
        4
        [[3. 0. 3. 2.]
         [2. 3. 3. 1.]]
        7
        [[1. 3. 3. 3. 0. 4. 3.]]
        2
        [[3. 2.]
         [1. 2.]]
        6
        [[4. 1. 4. 2. 1. 0.]
         [4. 4. 0. 4. 1. 2.]]
        8
        [[1. 1. 1. 1. 3. 4. 0. 4.]]

        :param array: 2D numpy array to be split
        :return: dictionary of numpy arrays
        """
        result = dict()
        for irow in range(matrix.shape[0]):
            splitted_arrays = utils.Utils.split_nan_array(matrix[irow, :])
            # filter out arrays that contain no information for markov process
            relevant_lists = list(filter(lambda x: True if len(x) > self.markov_order else False, splitted_arrays))
            for lst in relevant_lists:
                # check if all entries in the list are NaN
                if not np.all(np.isnan(lst)):
                    lst_length = len(lst)  # get key
                    try:
                        result[lst_length].append(lst)
                    except KeyError:
                        result[lst_length] = [lst]

        for k, v in result.items():
            # convert from list of arrays into numpy array
            v = np.array(v)
            # reassign to the same key
            result[k] = v
        return result

    def analyse_all_simulations(self, simulated_object):
        """
        Analyses whole simulated_object of shape (n_sim, n_rows, n_columns)
        sequentially calling disassemble_array(matrix) function on each cut simulated_object[i, :, :]
        It then merges all resulting dictionaries.

        :param array: 3D numpy array of shape (n_sim, n_rows, n_columns)
        :return: dictionary of numpy arrays
        """
        result = dict()
        for simulation_indx in range(simulated_object.shape[0]):
            # calculate on one simulation
            dict_arrays = self.disassemble_array(simulated_object[simulation_indx, :, :])
            # merge
            for panel_length, balanced_panel in dict_arrays.items():
                try:
                    concatenated = np.concatenate((result[panel_length], balanced_panel))
                    result[panel_length] = concatenated
                except KeyError:
                    result[panel_length] = balanced_panel
        return result

    def benchmarks(self, matrix):
        """
        calculates how many times in a row a number higer or equal to self.top
        (>= self.top) should show up in a matrix row in order to rule out chance
        with self.p confidence.

        :param array: 2D numpy matrix without mising entries
        :return: int. a number indicating how long a number >= self.top should show up
        """


if __name__ == "__main__":

    # small demo version of what this class is doing

    n_columns = 12  # time
    n_rows = 200  # number of observations
    drop_off = 0.3  # artificial drop off simulating missing entries in the panel
    n_sim = 100
    n_states = 100

    # generate some random data
    data = list()
    for i in range(n_columns):
        a = np.array(list(map(lambda x: float(x), range(n_rows))))
        a[np.random.randint(0, len(a), size=int(len(a) * drop_off))] = np.NaN
        np.random.shuffle(a)
        data.append(a)
    original_panel = np.array(data).T

    # generate transition probability matrix
    mask = np.ma.masked_invalid(original_panel).mask
    tpm_matrix = TPM.calculate_tpm(original_panel, n_states=n_states)
    # simulate
    sim = Simulator(tpm_matrix=tpm_matrix, mask=mask, n_sim=n_sim)
    sim_array = sim.simulate()

    # analyse simulation results
    start = time.time()
    a = SimulationAnalysis(original_panel)
    result = a.analyse_all_simulations(sim_array)
    for panel_size, panel in result.items():
        print()
        print(panel_size)
        print(panel)
    end = time.time()
    print("total:", end - start, "sec")