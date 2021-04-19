"""Containers and index orderings for the ODE models.

We're writing the ode systems under numba for speed. We'll statically define
some named tuples here that correspond to the parameters and COMPARTMENTS in
the ODE systems and then construct instances of those tuples to serve as
maps between semantically meaningful names and indices of the COMPARTMENTS
and parameters since numba plays nicely with named tuples.

"""
from collections import namedtuple
import os

import numpy as np


########################################
# Private static container definitions #
########################################

_Parameters = namedtuple(
    'Parameters', [
        'alpha', 'sigma', 'gamma1', 'gamma2', 'theta_minus',
        'rho_variant', 'pi', 'chi',
    ]
)

_FitParameters = namedtuple(
    'FitParameters', [
        'new_e', 'kappa', 'rho', 'phi',
    ]
)

_ForecastParameters = namedtuple(
    'ForecastParameters', [
        'beta_wild', 'beta_variant', 'theta_plus',
    ]
)


_NewE = namedtuple(
    'NewE', [
        'wild', 'variant_naive', 'variant_reinf', 'total'
    ]
)

_Aggregates = namedtuple(
    'Aggregates', [
        'susceptible_wild', 'susceptible_variant_only',
        'infectious_wild', 'infectious_variant',
        'n_total',
    ]
)

_VaccineTypes = namedtuple(
    'Vaccines', [
        'u', 'p', 'pa', 'm', 'ma',
    ]
)

_Compartments = namedtuple(
    'Compartments', [
        'S',            'E',            'I1',            'I2',            'R',
        'S_u',          'E_u',          'I1_u',          'I2_u',          'R_u',
        'S_p',          'E_p',          'I1_p',          'I2_p',          'R_p',
        'S_pa',         'E_pa',         'I1_pa',         'I2_pa',         'R_pa',

        'S_variant',    'E_variant',    'I1_variant',    'I2_variant',    'R_variant',
        'S_variant_u',  'E_variant_u',  'I1_variant_u',  'I2_variant_u',  'R_variant_u',
        'S_variant_pa', 'E_variant_pa', 'I1_variant_pa', 'I2_variant_pa', 'R_variant_pa',

        'S_m',                                                            'R_m',
    ]
)

_TrackingCompartments = namedtuple(
    'TrackingCompartments', [
        'NewE_wild', 'NewE_variant', 'NewE_p_wild', 'NewE_p_variant',
        'NewE_nbt', 'NewE_vbt', 'NewS_v', 'NewR_w',
        'V_u', 'V_p', 'V_pa', 'V_m', 'V_ma',
    ]
)

#####################
# Public index maps #
#####################

PARAMETERS = _Parameters(*list(range(len(_Parameters._fields))))
FIT_PARAMETERS = _FitParameters(*[
    i + len(PARAMETERS) for i in range(len(_FitParameters._fields))
])
FORECAST_PARAMETERS = _ForecastParameters(*[
    i + len(PARAMETERS) for i in range(len(_ForecastParameters._fields))
])
NEW_E = _NewE(*list(range(len(_NewE._fields))))
AGGREGATES = _Aggregates(*list(range(len(_Aggregates._fields))))
VACCINE_TYPES = _VaccineTypes(*list(range(len(_VaccineTypes._fields))))
COMPARTMENTS = _Compartments(*list(range(len(_Compartments._fields))))
TRACKING_COMPARTMENTS = _TrackingCompartments(*[
    i + len(COMPARTMENTS) for i in range(len(_TrackingCompartments._fields))
])

##############################
# Public component groupings #
##############################

UNVACCINATED = np.array([
    COMPARTMENTS.S,  COMPARTMENTS.S_variant,
    COMPARTMENTS.E,  COMPARTMENTS.E_variant,
    COMPARTMENTS.I1, COMPARTMENTS.I1_variant,
    COMPARTMENTS.I2, COMPARTMENTS.I2_variant,
    COMPARTMENTS.R,  COMPARTMENTS.R_variant,
])
SUSCEPTIBLE_WILD = np.array([
    COMPARTMENTS.S, COMPARTMENTS.S_u, COMPARTMENTS.S_p, COMPARTMENTS.S_pa,
])
SUSCEPTIBLE_VARIANT_ONLY = np.array([
    COMPARTMENTS.S_variant, COMPARTMENTS.S_variant_u, COMPARTMENTS.S_variant_pa, COMPARTMENTS.S_m,
])
INFECTIOUS_WILD = np.array([
    COMPARTMENTS.I1,    COMPARTMENTS.I2,
    COMPARTMENTS.I1_u,  COMPARTMENTS.I2_u,
    COMPARTMENTS.I1_p,  COMPARTMENTS.I2_p,
    COMPARTMENTS.I1_pa, COMPARTMENTS.I2_pa,
])
INFECTIOUS_VARIANT = np.array([
    COMPARTMENTS.I1_variant,    COMPARTMENTS.I2_variant,
    COMPARTMENTS.I1_variant_u,  COMPARTMENTS.I2_variant_u,
    COMPARTMENTS.I2_variant_pa, COMPARTMENTS.I2_variant_pa,
])


##########################
# Other public constants #
##########################

N_GROUPS = 2
# Turning off the JIT is operationally 1-to-1 with 
# saying something is broken in the ODE code and 
# I need to figure it out.
DEBUG = not bool(os.getenv('NUMBA_DISABLE_JIT', 0))

