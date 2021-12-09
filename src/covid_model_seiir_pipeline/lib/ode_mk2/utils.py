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


@numba.njit(cache=True)
def cartesian_product(arrays):
    """Generate a cartesian product of input arrays."""
    n = 1
    for x in arrays:
        n *= x.size
    out = np.zeros((n, len(arrays)))

    for i in range(len(arrays)):
        m = int(n / arrays[i].size)
        out[:n, i] = np.repeat(arrays[i], m)
        n //= arrays[i].size

    n = arrays[-1].size
    for k in range(len(arrays)-2, -1, -1):
        n *= arrays[k].size
        m = int(n / arrays[k].size)
        for j in range(1, arrays[k].size):
            out[j*m:(j+1)*m,k+1:] = out[0:m,k+1:]
    return out