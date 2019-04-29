#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Pipeline
--------

Contains methods calling all necessary functions sequentially.

"""
from marksim.tpm import TPM
from marksim.utils import utils
from marksim.simulate import Simulator
from marksim.analyse import SimulationAnalysis
from marksim.config import Configs as Co


def pipe(original_panel):
    """
    call functions sequentially

    :return: a dictionary with analysed simulation results
    """
    # generate transition probability matrix
    tpm_matrix = TPM.calculate_tpm(original_panel,
                                   n_states=Co.MARKOV_STATES,
                                   markov_order=Co.MARKOV_ORDER)
    # mask missing values (np.NaN)
    mask = np.ma.masked_invalid(original_panel).mask
    # simulate
    sim = Simulator(tpm_matrix=tpm_matrix,
                    mask=mask,
                    n_sim=Co.N_SIM)
    sim_array = sim.simulate()
    # convert original panel into states
    original_states_matrix = TPM.convert_to_states(original_panel, n_states=n_states)
    # analyse simulation results
    a = SimulationAnalysis(original_states_matrix,
                           top=Co.PERCENTILES,
                           p=Co.CONFIDENCE)

    return a.analyse_all_simulations(sim_array)
