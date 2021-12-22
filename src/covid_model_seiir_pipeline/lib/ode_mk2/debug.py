import os

import pandas as pd

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
    def _coerce(names, values) -> pd.Series:
        return pd.Series(values, index=names)

    @classmethod
    def compartments(cls, y):
        return cls._coerce(COMPARTMENTS_NAMES, y)

    @classmethod
    def tracking_compartments(cls, y):
        return cls._coerce(TRACKING_COMPARTMENTS_NAMES, y[len(COMPARTMENTS_NAMES):])

    @classmethod
    def parameters(cls, p):
        return cls._coerce(PARAMETERS_NAMES, p)

    @classmethod
    def new_e(cls, new_e):
        return cls._coerce(NEW_E_NAMES, new_e)

    @classmethod
    def effective_susceptible(cls, eff_s):
        return cls._coerce(EFFECTIVE_SUSCEPTIBLE_NAMES, eff_s)

    @classmethod
    def aggregates(cls, a):
        return cls._coerce(AGGREGATES_NAMES, a)
