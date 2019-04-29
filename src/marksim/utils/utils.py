#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Utils
-----
"""
import numpy as np


class Utils:
    """
    Utils is a collection of helper functions used in this package.
    """

    @staticmethod
    def split_nan_array(array):
        """
        Split numpy array in chunks using np.nan as separator

        >>> array
        [ 2. nan  4.  4.  3.  2.  0. nan  0. nan  1.  0.]
        >>> split_nan_array(array)
        [array([2.]), array([nan]), array([4., 4., 3., 2., 0.]), array([nan]), array([0.]), array([nan]), array([1., 0.])]

        :param array: numpy array to be split
        :return: list of numpy arrays
        """
        mask = np.ma.masked_invalid(array).mask
        return np.split(array, np.argwhere(np.diff(mask) != 0)[:, 0] + 1)
