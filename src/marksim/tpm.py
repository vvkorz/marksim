# -*- coding: utf-8  -*-
"""
Transition Probability Matrix
------------------------------

Module to calculate transition probability matrix.
"""
import numpy as np
import math
import numpy.ma as ma
from marksim.config import Configs


class TPM:
	"""
	TPM Class encapsulates all methods to calculate transition probability matrix of a discrete markov process
	"""
	
	def __init__(self):
		self.tpm_array = np.zeros(shape=(Configs.MARKOV_STATES,
										 Configs.MARKOV_STATES))
		self.markov_order = Configs.MARKOV_ORDER
		self.n_states = Configs.MARKOV_STATES

	def parse(self, array):
		"""
		This is a wrap up method to parse the array with unbalanced panel data and generate a transition probability matrix
		calling all necessary functions.

		:param array: 2D numpy array
		:return: transition probability matrix
		"""
		assert(len(array.shape) == 2),\
			"I only work with 2D arrays, not {} D".format(len(array.shape))
		assert(array.shape[0] > self.n_states),\
			"Number of states ({}) is bigger than number of rows {}. An instance can not be in many states simultaneously.".format(self.n_states, array.shape[0])

		self.tpm_array = TPM.calculate_tpm(array, n_states=self.n_states, markov_order=self.markov_order)

	@staticmethod
	def calculate_tpm(array, n_states=100, markov_order=1):
		"""
		Calculate a transition probability matrix of a descrete markov process given an unbalanced panel.

		The input array is treated as an unbalanced panel data. For example each row is a firm and each
		entry is a growth rate of this firm in a given period of time (columns). *np.nan* means a missing
		entry.

		The function:

          1. calculates states for each entry in a given array. For example, if the number of states is equal to 100, the function calculates in which percentile a given entry is.
          2. omits all transitions from a missing entry and to a missing entry. For example, *nan -> 0.* (see array[0, 0] & array[0,1] in the code below) or *5. -> nan* (see array[1, 0] & array[1,1] in the code below)
          3. calculates frequencies of transitions from one state to another and returns it as a transition probability matrix

        >>> data = list()
        >>> for i in range(10):  # generate unbalanced panel
        >>>     a = np.array(list(map(lambda x: float(x), range(10))))
        >>>     a[np.random.randint(0, len(a), size=int(len(a)*0.2))] = np.NaN
        >>>     np.random.shuffle(a)
        >>>     data.append(a)
        >>> array = np.array(data).T
        >>> print(array)    # randomly shuffled unbalanced panel
        [[nan  0.  1.  5.  9.  5. nan  8.  0.  4.]
         [ 5. nan  8.  9.  8.  0.  7.  9.  8.  0.]
         [ 8.  4.  9.  8. nan  3.  3.  1.  3.  1.]
         [ 9.  7.  7. nan  6.  8.  9. nan nan nan]
         [nan  2. nan nan  1.  4.  4.  3. nan  6.]
         [ 4.  5.  4.  7.  3.  9.  8.  5.  9.  7.]
         [ 6.  8.  6.  1.  4. nan  5.  4.  1.  3.]
         [ 3.  3.  0.  4. nan  1.  6.  7.  7.  8.]
         [ 7.  6.  3.  3.  5.  2.  2.  6.  4. nan]
         [ 1.  9. nan  6.  7.  7. nan nan  6.  9.]]
        >>> print(array.shape)
        (10, 10)
        >>> tpm = calculate_tpm(array, n_states=5)
        >>> tpm
        [[0.14285714 0.42857143 0.28571429 0.         0.14285714]
         [0.46153846 0.15384615 0.15384615 0.         0.23076923]
         [0.14285714 0.28571429 0.28571429 0.21428571 0.07142857]
         [0.25       0.25       0.25       0.         0.25      ]
         [0.         0.         0.16666667 0.83333333 0.        ]]
        >>> print(tpm.shape)
        (5, 5)

        :param array: 2D numpy array representing an unbalanced panel data
        :param n_states: number of states or bins in which the data should be splitted
        :param markov_order: order of the markov process
        :return: np.array() containing transition probability matrix
        """

		if array is None:
			return
		assert (isinstance(array, type(np.array((2, 2))))), "input must be {}, not {}".format(type(np.array((2, 2))),
																							  type(array))
		assert (len(array.shape) == 2), "array must be 2 dimentional, not {}".format(array.shape)

		# convert values into markov process states
		states_matrix = TPM.convert_to_states(array, n_states=n_states)
		# calculate frequencies
		all_possible_jumps = np.zeros(shape=(n_states, n_states))
		for index in range(states_matrix.shape[1] - 2):
			# collect all possible jumps
			g = states_matrix[:, index:index + 2]
			# filter all rows with NaN
			g = g[~np.isnan(g).any(axis=1)]
			unique_pairs, count_pairs = np.unique(g, axis=0, return_counts=True)
			# unique_pairs = tuple(map(lambda y: tuple([int(y[0]), int(y[1])]), unique_pairs))
			# inefficient code here
			for indx, key in enumerate(unique_pairs):
				all_possible_jumps[int(key[0]), int(key[1])] += count_pairs[indx]
		sum_rows = np.sum(all_possible_jumps, axis=1)
		# divide each row on its sum
		return np.divide(all_possible_jumps.T, sum_rows).T

	@staticmethod
	def f(x):
		"""
        a simple helper function

        >>> a = 10
        >>> b = [1,2,3]
        >>> x = (a, b)
        >>> f(x)
        [10, 10, 10]

        :param x: tuple of (int, list)
        :return: [int] * len(list)
        """
		return [x[0]] * len(x[1])

	@staticmethod
	def get_bins(array, bins=100):
		"""
		Calculate in which bin a given number of an array is. If bins=100 the function returns
		an array of percentiles where each number is a percentile

		>>> import numpy as np
		>>> array = np.array(range(20))
		>>> array
		[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
		>>> get_bins(array, bins=10)
		[0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9]
		>>> np.random.shuffle(array)  # shuffle the array
		>>> array
		[10 14  0  1 12  4  3 17 15  9  7 11 13  8 18  5 16  6  2 19]
		>>> get_bins(array, bins=10)
		[5 7 0 0 6 2 1 8 7 4 3 5 6 4 9 2 8 3 1 9]

		:param array: 1D numpy array. **must not contain NaNs**
		:param bins: number of states or bins in which the data should be splitted
		:return: np.array() preserving the order of elements
		"""
		if array is None:
			return

		assert (isinstance(array, type(np.array((1,))))), "input must be {}, not {}".format(type(np.array((1,))),
																							type(array))
		assert (len(
			array) > bins), "len(array) must be bigger than number of states. Given len(array)= {} , bins= {}".format(
			len(array), bins)
		sorted_indx = np.argsort(array)
		# split array into bins
		splitted_in_bins = np.array_split(sorted_indx, bins)
		# convert values in bins into states
		splitted_in_bins = np.array(list(map(lambda z: TPM.f(z), zip(range(bins), splitted_in_bins))))
		# flatten the array with states
		tiles = [item for sublist in splitted_in_bins for item in sublist]
		# create a dictionary with information on which state corresponds to which index
		states_dict = dict(zip(sorted_indx, tiles))
		# return an identical to input_array array of states
		return np.array(list(map(lambda z: states_dict[z], range(len(array)))))

	@staticmethod
	def calculate_states(array, n_states=100):
		"""
        Enhances a method get_bins by accepting an array with NaN values and ignoring them.
        It preserves the order of original array and returns NaN on the same place as in input.

        >>> import numpy as np
        >>> array = np.array(list(map(lambda x: float(x), range(20))))
        [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17. 18. 19.]
        >>> array[np.random.randint(0, len(array), size=int(len(array)*0.2))] = np.NaN
        [nan  1.  2. nan nan  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. nan 17. 18. 19.]
        >>> calculate_states(array, n_states=10)
        [nan  0.  0. nan nan  1.  1.  2.  2.  3.  3.  4.  4.  5.  5.  6. nan  7.  8.  9.]
        >>> np.random.shuffle(array)
        [ 8. nan 14.  5. nan nan 10. 13.  9. 18.  1. nan 11.  7. 12. 17.  2. 15. 6. 19.]
        >>> calculate_states(array, n_states=10)
        [ 2. nan  5.  1. nan nan  3.  5.  3.  8.  0. nan  4.  2.  4.  7.  0.  6. 1.  9.]

        :param array: 1D numpy array.
        :param n_states: number of states or bins in which the data should be split
        :return: np.array() preserving the order of elements
        """
		if array is None:
			return

		mask = np.invert(np.array(ma.masked_invalid(array).mask))
		shrinked_array = array[np.array(mask)]
		shrinked_array = TPM.get_bins(shrinked_array, bins=n_states)
		# expand array back to original size
		# extremely inefficient code here
		expanded_array = list()
		p = 0
		for indx in mask:
			if indx:
				expanded_array.append(shrinked_array[p])
				p += 1
			else:
				expanded_array.append(np.NaN)
		return np.array(expanded_array)

	@staticmethod
	def convert_to_states(array, n_states=100):
		"""
        converts a 2D numpy array into an identical 2D numpy array with values being replaced
        by a corresponding state, where states are calculated in accordance with calculate_states()
        function.

        >>> import numpy as np
        >>> data = list()
        >>> for i in range(10):
        >>>     a = np.array(list(map(lambda x: float(x), range(10))))
        >>>     a[np.random.randint(0, len(a), size=int(len(a)*0.2))] = np.NaN
        >>>     data.append(a)
        >>> array = np.array(data).T
        >>> print(array)  # very primitive unbalanced panel
        [[ 0. nan nan nan  0.  0. nan  0.  0.  0.]
         [ 1.  1. nan  1.  1. nan  1.  1.  1.  1.]
         [nan  2.  2.  2.  2.  2.  2.  2.  2.  2.]
         [ 3.  3.  3.  3.  3. nan nan  3. nan  3.]
         [nan  4.  4.  4.  4.  4.  4.  4.  4.  4.]
         [ 5.  5.  5.  5.  5.  5.  5. nan  5.  5.]
         [ 6.  6.  6. nan  6.  6.  6.  6. nan nan]
         [ 7.  7.  7.  7. nan  7.  7.  7.  7.  7.]
         [ 8.  8.  8.  8. nan  8.  8. nan  8.  8.]
         [ 9.  9.  9.  9.  9.  9.  9.  9.  9.  9.]]
        >>> array = convert_to_states(array, n_states=5)
        >>> print(array) # resulting convertion into 5 states or allocation in bins
        [[ 0. nan nan nan  0.  0. nan  0.  0.  0.]
         [ 0.  0. nan  0.  0. nan  0.  0.  0.  0.]
         [nan  0.  0.  0.  1.  0.  0.  1.  1.  1.]
         [ 1.  1.  0.  1.  1. nan nan  1. nan  1.]
         [nan  1.  1.  1.  2.  1.  1.  2.  1.  2.]
         [ 1.  2.  1.  2.  2.  1.  1. nan  2.  2.]
         [ 2.  2.  2. nan  3.  2.  2.  2. nan nan]
         [ 2.  3.  2.  2. nan  2.  2.  3.  2.  3.]
         [ 3.  3.  3.  3. nan  3.  3. nan  3.  3.]
         [ 4.  4.  4.  4.  4.  4.  4.  4.  4.  4.]]
        >>> data = list()
        >>> for i in range(10):
        >>>     a = np.array(list(map(lambda x: float(x), range(10))))
        >>>     a[np.random.randint(0, len(a), size=int(len(a)*0.2))] = np.NaN
        >>>     np.random.shuffle(a)
        >>>     data.append(a)
        >>> array = np.array(data).T
        >>> print(array)  # randomly shuffled unbalanced panel
        [[ 9.  6.  6.  2.  2.  1. nan nan  9.  7.]
         [nan  7. nan  8. nan  4.  5.  8.  8.  9.]
         [ 0.  1.  7. nan  8.  7. nan  6.  4.  0.]
         [ 3. nan  0. nan  4. nan  2.  4.  0.  1.]
         [ 7.  4.  4.  1.  0.  3.  8.  9. nan  2.]
         [ 8.  3.  8.  7.  6.  2.  0.  0. nan nan]
         [nan  9.  9.  9.  9.  0.  6.  1.  2. nan]
         [ 5.  2.  3.  5. nan  5.  9.  3.  1.  5.]
         [ 4. nan  2.  6.  7. nan  3.  7.  6.  4.]
         [ 1.  8.  1.  3.  3.  9.  7.  2.  3.  3.]]
        >>> array = convert_to_states(array, n_states=5)
        >>> print(array)  # resulting convertion into 5 states or allocation in bins
        [[ 4.  2.  2.  0.  0.  0. nan nan  4.  3.]
         [nan  2. nan  3. nan  2.  1.  3.  3.  4.]
         [ 0.  0.  3. nan  3.  3. nan  2.  2.  0.]
         [ 1. nan  0. nan  1. nan  0.  2.  0.  0.]
         [ 2.  1.  2.  0.  0.  1.  3.  4. nan  1.]
         [ 3.  1.  3.  2.  2.  1.  0.  0. nan nan]
         [nan  4.  4.  4.  4.  0.  2.  0.  1. nan]
         [ 2.  0.  1.  1. nan  2.  4.  1.  0.  2.]
         [ 1. nan  1.  2.  2. nan  1.  3.  2.  2.]
         [ 0.  3.  0.  1.  1.  4.  2.  1.  1.  1.]]


		:param array: 2D numpy array
		:param n_states: number o states/bins
		:return: array with states being calculates along columns, preserving np.NaNs
        """
		if array is None:
			return

		assert (isinstance(array, type(np.array((2, 2))))), "input must be {}, not {}".format(type(np.array((2, 2))),
																							  type(array))
		assert (len(array.shape) == 2), "array must be 2 dimensional, not {}".format(array.shape)

		return np.apply_along_axis(TPM.calculate_states, 0, array, n_states=n_states)

