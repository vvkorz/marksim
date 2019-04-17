# -*- coding: utf-8  -*-
"""
Module to calculate transition probability matrix.
"""
import numpy as np
from marksim.config import TPMConfigs


class TPM:
	"""
	Transition probability matrix of a markov process
	"""
	
	def __init__(self):
		self.tpm_array = np.zeros(shape=(TPMConfigs.MARKOV_STATES,
								         TPMConfigs.MARKOV_STATES))
		self.markov_order = TPMConfigs.MARKOV_ORDER
		self.n_states = TPMConfigs.MARKOV_STATES

	def parse(self, array):
		"""
		parse the array with unbalanced panel data and generate a transition probability matrix

		:param array: 2D numpy array
		:return:
		"""
		assert(len(array.shape) == 2),\
			"I only work with 2D arrays, not {} D".format(len(array.shape))
		assert(array.shape[0] > self.n_states),\
			"Number of states ({}) is bigger than number of rows {}. An instance can not be in many states simultaneously.".format(self.n_states, array.shape[0])

		# apply state calculation for each column
		states_array = np.apply_along_axis(self.calculate_states, 1, array)

	def calculate_states(self, array, num_st=TPMConfigs.MARKOV_STATES):
		"""
		A function returns a dictionary of states for every
		element in 1D array keeping missing values as missing values

		:param array: 1D array of data
		:param num_st:
		:return:
		"""
		assert(isinstance(array, type(np.array((1,))))), "input must be {}, not {}".format(type(np.array((1,))),
																						   type(array))

		# ignore all entries that are nan
		array = array[np.logical_not(np.isnan(array))]
		# array=list(filter(None.__ne__, array))
		array = sorted(array)
		states = dict()
		percent_step = int(len(array) / num_st) + 1
		print(percent_step)
		p = 1
		for i in range(0, len(array)):

			if i >= percent_step * p:
				p += 1
			states[array[i]] = p
		print(states)
		raise ValueError
		return states

