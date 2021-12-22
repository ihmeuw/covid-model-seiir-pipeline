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


@numba.njit
def cartesian_product(arrays):
    """Generate a cartesian product of input arrays."""
    n = 1
    for x in arrays:
        n *= len(x)
    out = np.zeros((n, len(arrays)), dtype=np.int64)

    for i in range(len(arrays)):
        m = int(n / len(arrays[i]))
        out[:n, i] = np.repeat(arrays[i], m)
        n //= len(arrays[i])

    n = len(arrays[-1])
    for k in range(len(arrays)-2, -1, -1):
        n *= len(arrays[k])
        m = int(n / len(arrays[k]))
        for j in range(1, len(arrays[k])):
            out[j*m:(j+1)*m,k+1:] = out[0:m,k+1:]
    return out