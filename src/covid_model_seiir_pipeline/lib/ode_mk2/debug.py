import os

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    PARAMETERS_NAMES,
    NEW_E_NAMES,
    EFFECTIVE_SUSCEPTIBLE_NAMES,
    COMPARTMENTS_NAMES,
    TRACKING_COMPARTMENTS_NAMES,
    AGGREGATES_NAMES,
)
# Turning off the JIT is operationally 1-to-1 with
# saying something is broken in the ODE code and
# I need to figure it out.
DEBUG = int(os.getenv('NUMBA_DISABLE_JIT', 0))


class Printer:

    @staticmethod
    def _print(names, values):
        for name, value in zip(names, values):
            print(f'{name:<30}: {value:>20.3f}')

    @classmethod
    def compartments(cls, y):
        cls._print(COMPARTMENTS_NAMES, y)

    @classmethod
    def tracking_compartments(cls, y):
        cls._print(TRACKING_COMPARTMENTS_NAMES, y[len(COMPARTMENTS_NAMES):])

    @classmethod
    def parameters(cls, p):
        cls._print(PARAMETERS_NAMES, p)

    @classmethod
    def new_e(cls, new_e):
        cls._print(NEW_E_NAMES, new_e)

    @classmethod
    def effective_susceptible(cls, eff_s):
        cls._print(EFFECTIVE_SUSCEPTIBLE_NAMES, eff_s)

    @classmethod
    def aggregates(cls, a):
        cls._print(AGGREGATES_NAMES, a)
