import numba
import numpy as np


##############################################
# Give indices semantically meaningful names #
##############################################
PARAMETERS = [
    'alpha', 'sigma', 'gamma1', 'gamma2', 'new_e',
    'u', 'p', 'pa', 'm', 'ma',
]
(
    alpha, sigma, gamma1, gamma2, new_e,
    u, p, pa, m, ma,
) = range(len(PARAMETERS))

COMPARTMENTS = [
    'S',    'E',    'I1',    'I2',    'R',
    'S_u',  'E_u',  'I1_u',  'I2_u',  'R_u',
    'S_p',  'E_p',  'I1_p',  'I2_p',  'R_p',
    'S_pa', 'E_pa', 'I1_pa', 'I2_pa', 'R_pa',
    'S_m',                            'R_m',
]
(
    s,    e,    i1,    i2,    r,
    s_u,  e_u,  i1_u,  i2_u,  r_u,
    s_p,  e_p,  i1_p,  i2_p,  r_p,
    s_pa, e_pa, i1_pa, i2_pa, r_pa,
    s_m,                      r_ma,
) = range(len(COMPARTMENTS))

_UNVACCINATED = np.array([s, e, i1, i2, r])
_SUSCEPTIBLE = np.array([s, s_u, s_p, s_pa])
_VACCINES = np.array([u, p, pa, m, ma])


@numba.njit
def system(t: float, y: np.ndarray, params: np.ndarray):
    n_unvaccinated = y[_UNVACCINATED].sum()
    total_susceptible = y[_SUSCEPTIBLE].sum()

    new_e_s = params[new_e] * y[s] / total_susceptible
    new_e_s_u = params[new_e] * y[s_u] / total_susceptible
    new_e_s_p = params[new_e] * y[s_p] / total_susceptible
    new_e_s_pa = params[new_e] * y[s_pa] / total_susceptible

    v_total = params[_VACCINES].sum()
    if v_total:
        expected_total_vaccines_s = v_total * y[s] / n_unvaccinated
        total_vaccines_s = min(y[s] - new_e_s, expected_total_vaccines_s)
        s_vaccines_u = total_vaccines_s * params[u] / v_total
        s_vaccines_p = total_vaccines_s * params[p] / v_total
        s_vaccines_pa = total_vaccines_s * params[pa] / v_total
        s_vaccines_m = total_vaccines_s * params[m] / v_total
        s_vaccines_ma = total_vaccines_s * params[ma] / v_total

        e_vaccines = min((1 - params[sigma]) * y[e],
                         v_total * y[e] / n_unvaccinated)
        i1_vaccines = min((1 - params[gamma1]) * y[i1],
                          v_total * y[i1] / n_unvaccinated)
        i2_vaccines = min((1 - params[gamma2]) * y[i2],
                          v_total * y[i2] / n_unvaccinated)
        r_vaccines = min(y[r], v_total * y[r] / n_unvaccinated)
    else:
        total_vaccines_s = 0.
        s_vaccines_u = 0.
        s_vaccines_p = 0.
        s_vaccines_pa = 0.
        s_vaccines_m = 0.
        s_vaccines_ma = 0.

        e_vaccines = 0.
        i1_vaccines = 0.
        i2_vaccines = 0.
        r_vaccines = 0.

    ds = -new_e_s - total_vaccines_s
    de = new_e_s - params[sigma]*y[e] - e_vaccines
    di1 = params[sigma]*y[e] - params[gamma1]*y[i1] - i1_vaccines
    di2 = params[gamma1]*y[i1] - params[gamma2]*y[i2] - i2_vaccines
    dr = params[gamma2]*y[i2] - r_vaccines

    ds_u = -new_e_s_u + s_vaccines_u
    de_u = new_e_s_u - sigma*y[e_u] + e_vaccines
    di1_u = params[sigma]*y[e_u] - params[gamma1]*y[i1_u] + i1_vaccines
    di2_u = params[gamma1]*y[i1_u] - params[gamma2]*y[i2_u] + i2_vaccines
    dr_u = params[gamma2]*y[i2_u] + r_vaccines

    ds_p = -new_e_s_p + s_vaccines_p
    de_p = new_e_s_p - sigma*y[e_p]
    di1_p = params[sigma]*y[e_p] - params[gamma1]*y[i1_p]
    di2_p = params[gamma1]*y[i1_p] - params[gamma2]*y[i2_p]
    dr_p = params[gamma2]*y[i2_p]

    ds_pa = -new_e_s_pa + s_vaccines_pa
    de_pa = new_e_s_pa - sigma*y[e_pa]
    di1_pa = params[sigma]*y[e_pa] - params[gamma1]*y[i1_pa]
    di2_pa = params[gamma1]*y[i1_pa] - params[gamma2]*y[i2_pa]
    dr_pa = params[gamma2]*y[i2_pa]

    ds_m = s_vaccines_m
    dr_m = s_vaccines_ma

    dy = np.array([
        ds, de, di1, di2, dr,
        ds_u, de_u, di1_u, di2_u, dr_u,
        ds_p, de_p, di1_p, di2_p, dr_p,
        ds_pa, de_pa, di1_pa, di2_pa, dr_pa,
        ds_m, dr_m,
    ])
    if dy.sum() > 1e-5:
        print('Compartment mismatch: ', dy.sum())

    return dy
