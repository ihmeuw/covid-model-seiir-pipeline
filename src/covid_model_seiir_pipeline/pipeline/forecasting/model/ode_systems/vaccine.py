import numba
import numpy as np

COMPARTMENTS = [
    'S',   'E',   'I1',   'I2',   'R',    # Unvaccinated
    'S_u', 'E_u', 'I1_u', 'I2_u', 'R_u',  # Vaccinated and unprotected
    'S_p', 'E_p', 'I1_p', 'I2_p', 'R_p',  # Vaccinated and protected
                                  'R_m',  # Vaccinated and immune
]
PARAMETERS = ['alpha', 'beta', 'sigma', 'gamma1', 'gamma2', 'p_immune', 'theta_plus', 'theta_minus']


@numba.njit
def system(t: float, y: np.ndarray, p: np.array):
    system_size = len(COMPARTMENTS)
    num_seiir_compartments = 5
    n_groups = y.size // system_size
    n_vaccines = 3 * n_groups
    p, vaccines = p[:-n_vaccines], p[-n_vaccines:]
    infectious = 0.
    n_total = y.sum()
    for i in range(n_groups):
        for j in range(3):  # Three sets of seiir compartments per group.
            # 3rd and 4th compartment of each group + seiir are infectious.
            infectious = (infectious
                          + y[i * system_size + j * num_seiir_compartments + 2]
                          + y[i * system_size + j * num_seiir_compartments + 3])

    dy = np.zeros_like(y)
    for i in range(n_groups):
        dy[i * system_size:(i + 1) * system_size] = single_group_system(
            t, y[i * system_size:(i + 1) * system_size], p, vaccines[i * 3:(i + 1) * 3], n_total, infectious
        )

    return dy


@numba.njit
def single_group_system(t: float, y: np.ndarray, p: np.ndarray,
                        vaccines: np.array, n_total: float, infectious: float):
    unvaccinated, unprotected, protected, r_m = y[:5], y[5:10], y[10:15], y[15]
    s, e, i1, i2, r = unvaccinated
    s_u, e_u, i1_u, i2_u, r_u = unprotected
    s_p, e_p, i1_p, i2_p, r_p = protected
    n_unvaccinated = unvaccinated.sum()

    alpha, beta, sigma, gamma1, gamma2, p_immune, theta_plus, theta_minus = p

    v_non_efficacious, v_protective, v_immune = vaccines
    v_efficacious = v_protective + v_immune
    if v_efficacious:
        p_immune = v_immune / v_efficacious
    else:
        p_immune = 0.

    v_total = v_non_efficacious + v_efficacious
    # Effective vaccines are efficacious vaccines delivered to
    # susceptible, unvaccinated individuals.
    v_effective = s / n_unvaccinated * v_efficacious
    # Some effective vaccines confer immunity, others just protect
    # from death after infection.
    v_immune = p_immune * v_effective
    v_protected = v_effective - v_immune

    # vaccinated and unprotected come from all bins.
    # Get count coming from S.
    v_unprotected_s = s / n_unvaccinated * v_non_efficacious

    # Expected vaccines coming out of S.
    s_vaccines = v_unprotected_s + v_protected + v_immune

    if s_vaccines:
        rho_unprotected = v_unprotected_s / s_vaccines
        rho_protected = v_protected / s_vaccines
        rho_immune = v_immune / s_vaccines
    else:
        rho_unprotected, rho_protected, rho_immune = 0, 0, 0
    # Actual count of vaccines coming out of S.
    s_vaccines = min(1 - beta * infectious ** alpha / n_total - theta_plus, s_vaccines / s) * s

    # Expected vaccines coming out of E.
    e_vaccines = e / n_unvaccinated * v_total
    # Actual vaccines coming out of E.
    e_vaccines = min(1 - sigma - theta_minus, e_vaccines / e) * e

    # Expected vaccines coming out of I1.
    i1_vaccines = i1 / n_unvaccinated * v_total
    # Actual vaccines coming out of I1.
    i1_vaccines = min(1 - gamma1, i1_vaccines / i1) * i1

    # Expected vaccines coming out of I2.
    i2_vaccines = i2 / n_unvaccinated * v_total
    # Actual vaccines coming out of I2.
    i2_vaccines = min(1 - gamma2, i2_vaccines / i2) * i2

    # Expected vaccines coming out of R.
    r_vaccines = r / n_unvaccinated * v_total
    # Actual vaccines coming out of R
    r_vaccines = min(1, r_vaccines / r) * r

    # Unvaccinated equations.
    # Normal Epi + vaccines causing exits from all compartments.
    new_e = beta * s * infectious ** alpha / n_total + theta_plus * s
    ds = -new_e - s_vaccines
    de = new_e - sigma * e - theta_minus * e - e_vaccines
    di1 = sigma * e - gamma1 * i1 - i1_vaccines
    di2 = gamma1 * i1 - gamma2 * i2 - i2_vaccines
    dr = gamma2 * i2 + theta_minus * e - r_vaccines

    # Vaccinated and unprotected equations
    # Normal epi + vaccines causing entrances to all compartments from
    # their unvaccinated counterparts.
    new_e_u = beta * s_u * infectious ** alpha / n_total + theta_plus * s_u
    ds_u = -new_e_u + rho_unprotected * s_vaccines
    de_u = new_e_u - sigma * e_u - theta_minus * e_u + e_vaccines
    di1_u = sigma * e_u - gamma1 * i1_u + i1_vaccines
    di2_u = gamma1 * i1_u - gamma2 * i2_u + i2_vaccines
    dr_u = gamma2 * i2_u + theta_minus * e_u + r_vaccines

    # Vaccinated and protected equations
    # Normal epi + protective vaccines taking people from S and putting
    # them in S_p
    new_e_p = beta * s_p * infectious ** alpha / n_total + theta_plus * s_p
    ds_p = -new_e_p + rho_protected * s_vaccines
    de_p = new_e_p - sigma * e_p - theta_minus * e_p
    di1_p = sigma * e_p - gamma1 * i1_p
    di2_p = gamma1 * i1_p - gamma2 * i2_p
    dr_p = gamma2 * i2_p + theta_minus * e_p

    # Vaccinated and immune
    dm = rho_immune * s_vaccines

    return np.array([
        ds, de, di1, di2, dr,
        ds_u, de_u, di1_u, di2_u, dr_u,
        ds_p, de_p, di1_p, di2_p, dr_p,
        dm
    ])
