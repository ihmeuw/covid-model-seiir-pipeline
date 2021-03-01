import numba
import numpy as np


##############################################
# Give indices semantically meaningful names #
##############################################
PARAMETERS = [
    'alpha', 'sigma', 'gamma1', 'gamma2',
    'new_e', 'm', 'u',
]
(
    alpha, sigma, gamma1, gamma2,
    new_e, m, u,
) = range(len(PARAMETERS))

COMPARTMENTS = [
    'S', 'E', 'I1', 'I2', 'R',
    'S_u', 'E_u', 'I1_u', 'I2_u', 'R_u',
    'R_m',
]
(
    s, e, i1, i2, r,
    s_u, e_u, i1_u, i2_u, r_u,
    r_m,
) = range(len(COMPARTMENTS))


@numba.njit
def system(t: float, y: np.ndarray, params: np.ndarray):
    n_unvaccinated = y[s] + y[e] + y[i1] + y[i2] + y[r]

    # Fraction of new infections coming from S vs S_u
    total_susceptible = y[s] + y[s_u]
    new_e_s = params[new_e] * y[s] / total_susceptible
    new_e_s_u = params[new_e] * y[s_u] / total_susceptible

    v_total = params[m] + params[u]
    if v_total:
        expected_total_vaccines_s = v_total * y[s] / n_unvaccinated
        total_vaccines_s = min(y[s] - new_e_s, expected_total_vaccines_s)
        s_vaccines_m = total_vaccines_s * params[m] / v_total
        s_vaccines_u = total_vaccines_s - s_vaccines_m

        e_vaccines = min((1 - params[sigma]) * y[e],
                         v_total * y[e] / n_unvaccinated)
        i1_vaccines = min((1 - params[gamma1]) * y[i1],
                          v_total * y[i1] / n_unvaccinated)
        i2_vaccines = min((1 - params[gamma2]) * y[i2],
                          v_total * y[i2] / n_unvaccinated)
        r_vaccines = min(y[r], v_total * y[r] / n_unvaccinated)
    else:
        s_vaccines_m = 0
        s_vaccines_u = 0
        e_vaccines = 0
        i1_vaccines = 0
        i2_vaccines = 0
        r_vaccines = 0

    ds = -new_e_s - s_vaccines_u - s_vaccines_m
    de = new_e_s - params[sigma]*y[e] - e_vaccines
    di1 = params[sigma]*y[e] - params[gamma1]*y[i1] - i1_vaccines
    di2 = params[gamma1]*y[i1] - params[gamma2]*y[i2] - i2_vaccines
    dr = params[gamma2]*y[i2] - r_vaccines

    ds_u = -new_e_s_u + s_vaccines_u
    de_u = new_e_s_u - sigma*y[e_u] + e_vaccines
    di1_u = params[sigma]*y[e_u] - params[gamma1]*y[i1_u] + i1_vaccines
    di2_u = params[gamma1]*y[i1_u] - params[gamma2]*y[i2_u] + i2_vaccines
    dr_u = params[gamma2]*y[i2_u] + r_vaccines
    
    dr_m = s_vaccines_m
    dy = np.array([
        ds, de, di1, di2, dr,
        ds_u, de_u, di1_u, di2_u, dr_u,
        dr_m,
    ])
    assert dy.sum() < 1e-5
    return dy
