#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Simulator
---------
"""


import numpy as np
from marksim.tpm import TPM
import array as arr
import time


class Simulator:
    """
    Simulator
    ---------

    Simulates markov process based on the transition probability array and a mask array that specifies
    how the result should look like.

    Let's say you have an unbalanced panel data that represent profits of 200 firms in the ast 12 years:

    >>> import numpy as np
    >>> from marksim import TPM
    >>> array
    [[ 89.  nan  70. ...  28. 167. 117.]
     [ nan  86. 157. ...  55. 196. 174.]
     [ nan 127.  nan ...  10. 185.  40.]
     ...
     [ 68.  97. 131. ...  nan  36.  95.]
     [ 48.  20. 194. ... 115. 187.  nan]
     [ 25.  23. 173. ... 173. 141. 191.]]

    A masked array obtained from this panel data shows which observations are missing:

    >>> mask = np.ma.masked_invalid(array).mask
    >>> mask
    [[False  True False ... False False False]
     [ True False False ... False False False]
     [ True False  True ... False False False]
     ...
     [False False False ...  True False False]
     [False False False ... False False  True]
     [False False False ... False False False]]

    If we imagine all these firms transiting from one percentile (thus n_states=100) to another, a transition probability
    matrix of a 1 order discrete markov process can be calculated as follows:

    >>> tpm_matrix = TPM.calculate_tpm(array, n_states=100)
    >>> tpm_matrix
    [[0.         0.         0.         ... 0.         0.         0.        ]
     [0.06666667 0.06666667 0.         ... 0.         0.06666667 0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]
     ...
     [0.         0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]]

    Now let's say we want to simulate a similar history twice (n_sim=2)

    >>> sim = Simulator(tpm_matrix=tpm_matrix, mask=mask, n_sim=2)
    >>> simualtions = sim.simulate()
    >>> simualtions
    [[[88. nan 39. ... 74. 30.  2.]
      [nan  7. 35. ... 60. 52. 44.]
      [nan 22. nan ... 29. 62. 45.]
      ...
      [46. 19. 12. ... nan 14. 44.]
      [46. 21. 52. ... 55. 38. nan]
      [57. 67. 53. ... 67. 33. 56.]]
     [[46. nan  8. ... 40. 30. 17.]
      [nan  4. 59. ... 39. 99. 40.]
      [nan 99. nan ... 20. 38. 47.]
      ...
      [22. 18.  4. ... nan 54. 74.]
      [22.  4. 32. ... 61. 23. nan]
      [62. 76. 20. ... 48.  8.  8.]]]
    >>> simualtions.shape
    (2, 200, 12)

    simulations.shape show (number_simulations, number_of_firms, number_of_years)

    """
    def __init__(self, tpm_matrix, mask=None, n_sim=1):
        self.tpm_matrix = tpm_matrix
        (self.panel_height, self.panel_length) = mask.shape
        self.mask = mask
        """a boolean numpy array of the shape of original panel"""
        self.n_sim = n_sim  # number of histories to simulate

    def walk(self):
        """
        Walks through markov chain given transition probability matrix.

        :return: np.array() of states
        """
        walk = np.array([0]*self.panel_length).astype(int)
        states = np.array(range(self.tpm_matrix.shape[1])).astype(int)
        for i, state in enumerate(walk):
            # random number from
            next_state = np.random.choice(states, size=1, p=self.tpm_matrix[walk[i-1], :])[0]
            walk[i] = next_state
        return np.array(walk)

    def simulate_history(self):
        """
        Simulates history once given

         - transition probability matrix of a discrete markov process
         - mask, specifies which entries should be np.NaN and size of the panel

        :return: np.array() a copy of the shape of panel size.
        """
        history = tuple(map(lambda x: self.walk(), range(self.panel_height)))
        if self.mask is not None:
            history = np.ma.masked_where(self.mask, history)
            history = history.astype(float, copy=False)
        return np.array(history.filled(np.nan))

    def simulate(self):
        """
        Simulates histories. The final array will contain states that were visited
        during walks over markov chain

        :return: np.array() shape(n_sim, n_rows, n_columns)
        """
        result = list()
        for j in range(self.n_sim):
            result.append(self.simulate_history())
        return np.array(result)


if __name__ == "__main__":

    # small demo version of what this class is doing
    start = time.time()
    n_columns = 12  # time
    n_rows = 200  # number of observations
    drop_off = 0.3
    n_sim = 2
    n_states = 100

    # generate some random data
    data = list()
    for i in range(n_columns):
        a = np.array(list(map(lambda x: float(x), range(n_rows))))
        a[np.random.randint(0, len(a), size=int(len(a)*drop_off))] = np.NaN
        np.random.shuffle(a)
        data.append(a)
    array = np.array(data).T
    print("initial data:")
    print(array)
    print()
    mask = np.ma.masked_invalid(array).mask
    print("mask:")
    print(mask)
    print()
    tpm_matrix = TPM.calculate_tpm(array, n_states=n_states)
    print("transition probability matrix:")
    print(tpm_matrix)
    print()
    print("timing simulation process")
    sim = Simulator(tpm_matrix=tpm_matrix, mask=mask, n_sim=n_sim)
    sim_array = sim.simulate()

    end = time.time()
    print("total:", end - start, "sec")
    print()
    print('simulations object shape', sim_array.shape)
    print("simulated object")
    print(sim_array)
    print()
