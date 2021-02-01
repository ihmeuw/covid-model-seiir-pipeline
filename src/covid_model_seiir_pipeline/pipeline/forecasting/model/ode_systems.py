import numba
import numpy as np


@numba.njit
def seiir_single_group_system(t: float, y: np.ndarray, p: np.ndarray, n_total: float, infectious: float):
    s, e, i1, i2, r = y
    alpha, sigma, gamma1, gamma2, beta, theta_plus, theta_minus = p

    new_e = beta * (s / n_total) * infectious ** alpha

    ds = -new_e - theta_plus * s
    de = new_e + theta_plus * s - sigma * e - theta_minus * e
    di1 = sigma * e - gamma1 * i1
    di2 = gamma1 * i1 - gamma2 * i2
    dr = gamma2 * i2 + theta_minus * e

    return np.array([ds, de, di1, di2, dr])


@numba.njit
def seiir_system(t: float, y: np.ndarray, p: np.array):
    system_size = 5
    n_groups = y.size // system_size
    infectious = 0.
    n_total = y.sum()
    for i in range(n_groups):
        # 3rd and 4th compartment of each group are infectious.
        infectious = infectious + y[i * system_size + 2] + y[i * system_size + 3]

    dy = np.zeros_like(y)
    for i in range(n_groups):
        dy[i * system_size:(i + 1) * system_size] = seiir_single_group_system(
            t, y[i * system_size:(i + 1) * system_size], p, n_total, infectious,
        )

    return dy


@numba.njit
def vaccine_single_group_system(t: float, y: np.ndarray, p: np.ndarray,
                                vaccines: np.array, n_total: float, infectious: float):
    unvaccinated, unprotected, protected, m = y[:5], y[5:10], y[10:15], y[15]
    s, e, i1, i2, r = unvaccinated
    s_u, e_u, i1_u, i2_u, r_u = unprotected
    s_p, e_p, i1_p, i2_p, r_p = protected
    n_unvaccinated = unvaccinated.sum()

    alpha, sigma, gamma1, gamma2, p_immune, beta, theta_plus, theta_minus = p

    v_non_efficacious, v_efficacious = vaccines

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


@numba.njit
def vaccine_system(t: float, y: np.ndarray, p: np.array):
    system_size = 16
    num_seiir_compartments = 5
    n_groups = y.size // system_size
    n_vaccines = 2 * n_groups
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
        dy[i * system_size:(i + 1) * system_size] = vaccine_single_group_system(
            t, y[i * system_size:(i + 1) * system_size], p, vaccines[i * 2:(i + 1) * 2], n_total, infectious
        )

    return dy


@numba.njit
def get_vaccines_out(y, vaccines, params, b_wild, b_variant):
    s, e, i1, i2, r = 0, 1, 2, 3, 4
    e_variant, i1_variant, i2_variant, r_variant = 22, 23, 24, 25

    alpha, sigma, gamma1, gamma2 = 0, 1, 2, 3
    theta_plus, theta_minus = 6, 7

    u, p, pa, m, ma = 0, 1, 2, 3, 4

    vaccines_out = np.zeros(y.size, vaccines.size)

    n_unvaccinated = y[s, e, i1, i2, r, e_variant, i1_variant, i2_variant, r_variant].sum()
    v_total = vaccines.sum()

    if n_unvaccinated:
        # Folks in S can have all vaccine outcomes
        # Compute vaccines coming from S
        new_e_wild_from_s = b_wild * y[s] + params[theta_plus] * y[s]
        new_e_variant_from_s = b_variant * y[s]
        new_e_total_from_s = new_e_wild_from_s + new_e_variant_from_s
        new_e_variant_from_r = b_variant * y[r]

        expected_total_vaccines_s = y[s] / n_unvaccinated * v_total
        total_vaccines_s = min(y[s] - new_e_total_from_s, expected_total_vaccines_s)
        if expected_total_vaccines_s:
            for vaccine_type in [u, p, pa, m, ma]:
                expected_vaccines = y[s] / n_unvaccinated * vaccines[vaccine_type]
                rho = expected_vaccines / expected_total_vaccines_s
                vaccines_out[s, vaccine_type] = rho * total_vaccines_s

        # Folks in E, I2, I2 only unprotected
        vaccines_out[e, u] = min((1 - params[sigma] - params[theta_minus]) * y[e],
                                 y[e] / n_unvaccinated * v_total)
        vaccines_out[i1, u] = min((1 - params[gamma1]) * y[i1],
                                  y[i1] / n_unvaccinated * v_total)
        vaccines_out[i2, u] = min((1 - params[gamma2]) * y[i2],
                                  y[i2] / n_unvaccinated * v_total)

        # Folks in R can be protected or immunized from the variant
        expected_total_vaccines_r = y[r] / n_unvaccinated * v_total
        total_vaccines_r = min(y[r] - new_e_variant_from_r, expected_total_vaccines_r)
        if expected_total_vaccines_r:
            expected_u_vaccines = y[r] / n_unvaccinated * vaccines[u, p, m].sum()
            rho = expected_u_vaccines / expected_total_vaccines_r
            vaccines_out[r, u] = rho * total_vaccines_r
            for vaccine_type in [pa, ma]:
                expected_vaccines = y[r] / n_unvaccinated * vaccines[vaccine_type]
                rho = expected_vaccines / expected_total_vaccines_r
                vaccines_out[r, vaccine_type] = rho * total_vaccines_r

        vaccines_out[e_variant, u] = min((1 - params[sigma]) * y[e_variant],
                                         y[e_variant] / n_unvaccinated * v_total)
        vaccines_out[i1_variant, u] = min((1 - params[gamma1]) * y[i1_variant],
                                          y[i1_variant] / n_unvaccinated * v_total)
        vaccines_out[i2_variant, u] = min((1 - params[gamma2]) * y[i2_variant],
                                          y[i2_variant] / n_unvaccinated * v_total)
        vaccines_out[r_variant, u] = min(y[r_variant], y[r_variant] / n_unvaccinated * v_total)


@numba.njit
def variant_single_group_system(t: float,
                                y: np.ndarray, params: np.ndarray,
                                b_wild: float, b_variant: float, b_variant_r: float,
                                vaccines: np.ndarray):
    s, e, i1, i2, r = 0, 1, 2, 3, 4
    s_u, e_u, i1_u, i2_u, r_u = 5, 6, 7, 8, 9
    s_p, e_p, i1_p, i2_p, r_p = 10, 11, 12, 13, 14
    s_pa, e_pa, i1_pa, i2_pa, r_pa = 15, 16, 17, 18, 19
    r_m, r_ma = 20, 21

    e_variant, i1_variant, i2_variant, r_variant = 22, 23, 24, 25
    e_u_variant, i1_u_variant, i2_u_variant, r_u_variant = 26, 27, 28, 29
    e_pa_variant, i1_pa_variant, i2_pa_variant, r_pa_variant = 30, 31, 32, 33

    alpha, sigma, gamma1, gamma2 = 0, 1, 2, 3
    theta_plus, theta_minus = 6, 7

    u, p, pa, m, ma = 0, 1, 2, 3, 4

    vaccines_out = get_vaccines_out(y, vaccines, params, b_wild, b_variant)

    outflow_map = {
        # Unvaccinated
        s: [
            (e, b_wild + params[theta_plus], 0.),
            (s_u, 0., vaccines_out[s, u]),
            (s_p, 0., vaccines_out[s, p]),
            (s_pa, 0., vaccines_out[s, pa]),
            (r_m, 0., vaccines_out[s, m]),
            (r_ma, 0., vaccines_out[s, ma]),
            (e_variant, b_variant, 0.),
        ],
        e: [
            (i1, params[sigma], 0.),
            (r, params[theta_minus], 0.),
            (e_u, 0., vaccines_out[e, u]),
        ],
        i1: [
            (i2, gamma1, 0.),
            (i1_u, 0., vaccines_out[i1, u]),
        ],
        i2: [
            (r, gamma2, 0.),
            (i2_u, 0., vaccines_out[i2, u]),
        ],
        r: [
            (r_u,  0., vaccines_out[r, u]),
            (r_pa, 0., vaccines_out[r, pa]),
            (r_ma, 0., vaccines_out[r, ma]),
            (e_variant, b_variant_r, 0.),
        ],
        # Unprotected
        s_u: [
            (e_u, b_wild + params[theta_plus], 0.),
            (e_u_variant, b_variant, 0.),
        ],
        e_u: [
            (i1_u, params[sigma], 0.),
            (r_u, params[theta_minus], 0.),
        ],
        i1_u: [
            (i2_u, params[gamma1], 0.),
        ],
        i2_u: [
            (r_u, params[gamma2], 0.),
        ],
        r_u: [
            (e_u_variant, b_variant_r, 0.),
        ],
        # Protected from wild-type
        s_p: [
            (e_p, b_wild + params[theta_plus], 0.),
            (e_u_variant, b_variant, 0.),
        ],
        e_p: [
            (i1_p, params[sigma], 0.),
            (r_p, params[theta_minus], 0.)
        ],
        i1_p: [
            (i2_p, params[gamma1], 0.),
        ],
        i2_p: [
            (r_p, params[gamma2], 0.),
        ],
        r_p: [
            (e_u_variant, b_variant_r, 0.)
        ],
        # Protected from all-types
        s_pa: [
            (e_pa, b_wild + params[theta_plus], 0.),
            (e_pa_variant, b_variant, 0.),
        ],
        e_pa: [
            (i1_pa, params[sigma], 0.),
            (r_pa, params[theta_minus], 0.),
        ],
        i1_pa: [
            (i2_pa, params[gamma1], 0.),
        ],
        i2_pa: [
            (r_pa, params[gamma2], 0.),
        ],
        r_pa: [
            (e_pa_variant, b_variant_r, 0.),
        ],
        # Immune
        r_m: [
            (e_u_variant, b_variant_r, 0.),  # FIXME: does this go to e_pa_variant?
        ],
        r_ma: [],
        # Unvaccinated variant
        e_variant: [
            (i1_variant, params[sigma], 0.),
            (e_u_variant, 0., vaccines_out[e_variant, u]),
        ],
        i1_variant: [
            (i2_variant, params[gamma1], 0.),
            (i1_u_variant, 0., vaccines_out[i1_variant, u])
        ],
        i2_variant: [
            (r_variant, params[gamma2], 0.),
            (i2_u_variant, 0., vaccines_out[i2_u_variant, u]),
        ],
        r_variant: [
            (r_u_variant, 0., vaccines_out[r_variant, u]),
        ],
        # Unprotected variant
        e_u_variant: [
            (i1_u_variant, params[sigma], 0.),
        ],
        i1_u_variant: [
            (i2_u_variant, params[gamma1], 0.),
        ],
        i2_u_variant: [
            (r_u_variant, params[gamma2], 0.),
        ],
        r_u_variant: [],
        # Protected variant
        e_pa_variant: [
            (i1_pa_variant, params[sigma], 0.),
        ],
        i1_pa_variant: [
            (i2_pa_variant, params[gamma1], 0.),
        ],
        i2_pa_variant: [
            (r_pa_variant, params[gamma2], 0.),
        ],
    }

    result = np.zeros_like(y)
    for out_compartment, outflows in outflow_map.items():
        for in_compartment, relative_change, absolute_change in outflows:
            result[in_compartment] = relative_change * y[out_compartment] + absolute_change

    assert result.sum() < 1e-10, 'Compartment mismatch'

    return result


@numba.njit
def variant_natural_single_group_system(t: float, y: np.ndarray, params: np.ndarray,
                                        vaccines: np.ndarray, n_total: float, infectious: np.ndarray):
    alpha, sigma, gamma1, gamma2 = 0, 1, 2, 3
    beta, beta_variant = 4, 5

    infectious_wild, infectious_variant = infectious
    b_wild = params[beta] * infectious_wild ** params[alpha] / n_total
    b_variant = params[beta_variant] * infectious_variant ** params[alpha] / n_total
    b_variant_r = b_variant
    return variant_single_group_system(t, y, params, b_wild, b_variant, b_variant_r, vaccines)


@numba.njit
def variant_implicit_single_group_system(t: float, y: np.ndarray, params: np.ndarray,
                                         vaccines: np.ndarray, n_total: float, infectious: np.ndarray):

    alpha, sigma, gamma1, gamma2 = 0, 1, 2, 3
    beta, beta_variant = 4, 5
    variant_prevalence = 8

    i_total = infectious.sum()
    b_wild = params[beta] * (1 - params[variant_prevalence]) * i_total ** params[alpha] / n_total
    b_variant = params[beta_variant] * params[variant_prevalence] * i_total ** params[alpha] / n_total
    b_variant_r = b_variant
    return variant_single_group_system(t, y, params, b_wild, b_variant, b_variant_r, vaccines)


@numba.njit
def variant_explicit_single_group_system(t: float, y: np.ndarray, params: np.ndarray,
                                         vaccines: np.ndarray, n_total: float, infectious: np.ndarray):
    alpha, sigma, gamma1, gamma2 = 0, 1, 2, 3
    beta, beta_variant = 4, 5
    variant_prevalence = 8

    infectious_wild, infectious_variant = infectious
    b_combined = (params[beta] * infectious_wild ** params[alpha] / n_total
                  + params[beta_variant] * infectious_variant ** params[alpha] / n_total)
    b_wild = (1 - params[variant_prevalence]) * b_combined
    b_variant = params[variant_prevalence] * b_combined
    b_variant_r = params[variant_prevalence] * params[beta_variant] * infectious_variant ** params[alpha] / n_total

    return variant_single_group_system(t, y, params, b_wild, b_variant, b_variant_r, vaccines)


@numba.njit
def variant_system(t: float, y: np.ndarray, p: np.ndarray, single_group_system):
    system_size = 34
    n_groups = y.size // system_size
    n_vaccines = 4 * n_groups
    p, vaccines = p[:-n_vaccines], p[-n_vaccines:]
    infectious_wild = 0.
    infectious_variant = 0.
    n_total = y.sum()

    infectious_wild_indices = [2, 3, 7, 8, 12, 13, 17, 18]
    for i in infectious_wild_indices:
        for j in range(n_groups):
            infectious_wild = infectious_wild + y[j * system_size + i]

    infectious_variant_indices = [23, 24, 27, 28, 31, 32]
    for i in infectious_variant_indices:
        for j in range(n_groups):
            infectious_variant = infectious_variant + y[j * system_size + i]

    infectious = np.array([infectious_wild, infectious_variant])

    dy = np.zeros_like(y)
    for i in range(n_groups):
        dy[i * system_size:(i + 1) * system_size] = single_group_system(
            t, y[i * system_size:(i + 1) * system_size], p, vaccines[i * 2:(i + 1) * 2], n_total, infectious
        )

    return dy


@numba.njit
def variant_natural_system(t: float, y: np.ndarray, p: np.ndarray):
    return variant_system(t, y, p, variant_natural_single_group_system)


@numba.njit
def variant_implicit_system(t: float, y: np.ndarray, p: np.ndarray):
    return variant_system(t, y, p, variant_implicit_single_group_system)


@numba.njit
def variant_explicit_system(t: float, y: np.ndarray, p: np.ndarray):
    return variant_system(t, y, p, variant_explicit_single_group_system)
