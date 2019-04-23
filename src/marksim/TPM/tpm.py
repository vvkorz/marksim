# -*- coding: utf-8  -*-
"""
Module to calculate transition probability matrix.
"""
import numpy as np
import math
import numpy.ma as ma
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
		:return: transition probability matrix
		"""
		assert(len(array.shape) == 2),\
			"I only work with 2D arrays, not {} D".format(len(array.shape))
		assert(array.shape[0] > self.n_states),\
			"Number of states ({}) is bigger than number of rows {}. An instance can not be in many states simultaneously.".format(self.n_states, array.shape[0])

		states_array = TPM.convert_to_states(array, n_states=5)

	@staticmethod
	def f(x):
		"""
        simple function

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

        >>> data = list()
        >>> for i in range(10):
        >>>     a = np.array(list(map(lambda x: float(x), range(10))))
        >>>     a[np.random.randint(0, len(a), size=int(len(a)*0.2))] = np.NaN
        >>>     data.append(a)
        >>> array = np.array(data).T
        >>> print(array)
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
        >>> print(array)
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
        >>> print(array)
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
        >>> print(array)
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
		assert (isinstance(array, type(np.array((2, 2))))), "input must be {}, not {}".format(type(np.array((2, 2))),
																							  type(array))
		assert (len(array.shape) == 2), "array must be 2 dimensional, not {}".format(array.shape)

		return np.apply_along_axis(TPM.calculate_states, 0, array, n_states=n_states)

