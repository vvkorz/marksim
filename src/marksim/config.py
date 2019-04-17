# -*- coding: utf-8  -*-
"""
Specifies the set of configurations for a marksim package
"""
import os


class TPMConfigs:
    """
    Configurations
    """

    MARKOV_ORDER = 1
    """the order of a markov property or how memoryless is our simulation"""
    MARKOV_STATES = 100
    """number of states of the markov process"""
    TPM_SAVE_PATH = os.path.dirname(os.path.abspath(__file__))
    """path to save transition probability matrixes and panel structure objects (pickled objects)"""


class SimConfigs:
    """
    Configurations of the simulations module
    """

    N_SIM = 100
    """number of simulations to perform"""
    SIM_SAVE_PATH = os.path.dirname(os.path.abspath(__file__))
    """path to save simulations (pickled objects)"""


class AnalyticsConfigs:
    """
    Configurations of the analytics module
    """

    LATEX = False
    """generate code for latex tables"""
    RESULTS_SAVE_PATH = os.path.dirname(os.path.abspath(__file__))
    """path to save results (JSON objects and latex code)"""
    PERCENTILES = 10
    """whether to analyse top 10 or 20 or etc... """
