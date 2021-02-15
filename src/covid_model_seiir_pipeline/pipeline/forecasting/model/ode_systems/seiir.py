import numba
import numpy as np


COMPARTMENTS = ['S', 'E', 'I1', 'I2', 'R']
PARAMETERS = ['alpha', 'beta', 'sigma', 'gamma1', 'gamma2', 'theta_plus', 'theta_minus']


@numba.njit
def system(t: float, y: np.ndarray, p: np.array):
    system_size = 5
    n_groups = y.size // system_size
    infectious = 0.
    n_total = y.sum()
    for i in range(n_groups):
        # 3rd and 4th compartment of each group are infectious.
        infectious = infectious + y[i * system_size + 2] + y[i * system_size + 3]

    dy = np.zeros_like(y)
    for i in range(n_groups):
        dy[i * system_size:(i + 1) * system_size] = single_group_system(
            t, y[i * system_size:(i + 1) * system_size], p, n_total, infectious
        )

    return dy


@numba.njit
def single_group_system(t: float, y: np.ndarray, p: np.ndarray, n_total: float, infectious: float):
    s, e, i1, i2, r = y
    alpha, beta, sigma, gamma1, gamma2, theta_plus, theta_minus = p

    new_e = beta * (s / n_total) * infectious ** alpha

    ds = -new_e - theta_plus * s
    de = new_e + theta_plus * s - sigma * e - theta_minus * e
    di1 = sigma * e - gamma1 * i1
    di2 = gamma1 * i1 - gamma2 * i2
    dr = gamma2 * i2 + theta_minus * e

    return np.array([ds, de, di1, di2, dr])
