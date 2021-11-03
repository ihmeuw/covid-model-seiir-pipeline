import numba
import numpy as np

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    RISK_GROUP,
)


@numba.njit
def safe_divide(a: float, b: float):
    """Divide that returns zero if numerator and denominator are both zero."""
    if b == 0.0:
        assert a == 0.0
        return 0.0
    return a / b


@numba.njit
def subset_risk_group(x: np.ndarray, risk_group: int):
    x_size = x.size // len(RISK_GROUP)
    group_start = risk_group * x_size
    group_end = (risk_group + 1) * x_size
    return x[group_start:group_end]
