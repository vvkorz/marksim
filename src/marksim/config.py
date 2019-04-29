# -*- coding: utf-8  -*-
"""
Configurations
--------------

Specifies the set of configurations for a marksim package
"""
import os


class Configs:
    """
    Configurations

    """
    # TODO This might be not the best design decision to pack all variables under one class.
    # tpm configs
    MARKOV_ORDER = 1
    """the order of a markov property or how memoryless is our simulation"""
    MARKOV_STATES = 100
    """number of states of the markov process"""

    # simulation configs
    N_SIM = 100
    """number of simulations to perform"""

    # analysis configs
    PERCENTILES = 80
    """whether to analyse top 10 or 20 or etc... """
    CONFIDENCE = 99
    """p value"""
