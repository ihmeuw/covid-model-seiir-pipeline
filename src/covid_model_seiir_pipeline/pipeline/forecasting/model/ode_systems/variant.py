import itertools

import numba
import numpy as np


##############################################
# Give indices semantically meaningful names #
##############################################
# Parameters
PARAMETERS = [
    'alpha', 'beta_wild', 'beta_variant', 'sigma', 'gamma1', 'gamma2',
    'theta_plus', 'theta_minus',
    'p_wild', 'p_variant',
    'p_cross_immune',
]
(
    alpha, beta_wild, beta_variant, sigma, gamma1, gamma2,
    theta_plus, theta_minus,
    p_wild, p_variant,
    p_cross_immune
) = range(len(PARAMETERS))

# Compartments
SEIIR_COMPARTMENTS = ['S', 'E', 'I1', 'I2', 'R']
WILD_TYPE_GROUPS = ['', '_u', '_p', '_pa']
VARIANT_TYPE_GROUPS = ['', '_u', '_pa']

WILD_TYPE_COMPARTMENTS = [f'{compartment}{group}' for group, compartment
                          in itertools.product(WILD_TYPE_GROUPS, SEIIR_COMPARTMENTS)]
VARIANT_TYPE_COMPARTMENTS = [f'{compartment}_variant{group}' for group, compartment
                             in itertools.product(VARIANT_TYPE_GROUPS, SEIIR_COMPARTMENTS)]
IMMUNE_COMPARTMENTS = ['S_m', 'R_m']
TRACKING_COMPARTMENTS = [
    'NewE_wild', 'NewE_variant', 'NewE_p_wild', 'NewE_p_variant',
    'NewE_nbt', 'NewE_vbt', 'NewS_v', 'NewR_w',
    'V_u', 'V_p', 'V_pa', 'V_m', 'V_ma',
]
COMPARTMENTS = WILD_TYPE_COMPARTMENTS + VARIANT_TYPE_COMPARTMENTS + IMMUNE_COMPARTMENTS + TRACKING_COMPARTMENTS

(
    # Wild type compartments
    s, e, i1, i2, r,
    s_u, e_u, i1_u, i2_u, r_u,
    s_p, e_p, i1_p, i2_p, r_p,
    s_pa, e_pa, i1_pa, i2_pa, r_pa,
    # Variant type compartments
    s_variant, e_variant, i1_variant, i2_variant, r_variant,
    s_u_variant, e_u_variant, i1_u_variant, i2_u_variant, r_u_variant,
    s_pa_variant, e_pa_variant, i1_pa_variant, i2_pa_variant, r_pa_variant,
    # Immune compartments
    s_m, r_ma,
    # Composite tracking compartments.
    new_e_wild, new_e_variant, new_e_p_wild, new_e_p_variant,
    new_e_nbt, new_e_vbt, new_s_v, new_r_w,
    v_u, v_p, v_pa, v_m, v_ma,
) = range(len(COMPARTMENTS))

# Vaccine categories.
VACCINE_CATEGORIES = [
    'u', 'p', 'pa', 'm', 'ma'
]
(
    u, p, pa, m, ma
) = range(len(VACCINE_CATEGORIES))


N_SEIIR_COMPARTMENTS = len(SEIIR_COMPARTMENTS)
N_WILD_TYPE_SEIIR_GROUPS = len(WILD_TYPE_GROUPS)
N_VARIANT_TYPE_SEIIR_GROUPS = len(VARIANT_TYPE_GROUPS)
N_IMMUNE_COMPARTMENTS = len(IMMUNE_COMPARTMENTS)
N_TRACKING_COMPARTMENTS = len(TRACKING_COMPARTMENTS)
GROUP_SYSTEM_SIZE = len(COMPARTMENTS)
REAL_SYSTEM_SIZE = GROUP_SYSTEM_SIZE - N_TRACKING_COMPARTMENTS

N_GROUPS = 2
N_VACCINE_CATEGORIES = 5
N_VACCINE_PARAMETERS = N_VACCINE_CATEGORIES * N_GROUPS

# 3rd and 4th compartment of each seiir group are infectious.
LOCAL_I1 = 2
LOCAL_I2 = 3


# TODO: We're one layer of abstraction deeper than necessary.
@numba.njit
def variant_natural_system(t: float, y: np.ndarray, params: np.ndarray):
    return variant_system(t, y, params, variant_natural_single_group_system)


@numba.njit
def variant_implicit_system(t: float, y: np.ndarray, params: np.ndarray):
    return variant_system(t, y, params, variant_implicit_single_group_system)


@numba.njit
def variant_explicit_system(t: float, y: np.ndarray, params: np.ndarray):
    return variant_system(t, y, params, variant_explicit_single_group_system)


@numba.njit
def variant_system(t: float, y: np.ndarray, params: np.ndarray, single_group_system):
    # Split parameters from vaccines
    params, vaccines = params[:-N_VACCINE_PARAMETERS], params[-N_VACCINE_PARAMETERS:]

    # Demographic groups mix, so we need to precompute infectious folks.
    infectious_wild = 0.
    infectious_variant = 0.
    n_total = y[:REAL_SYSTEM_SIZE].sum()
    for i in range(N_GROUPS):
        for j in range(N_WILD_TYPE_SEIIR_GROUPS):
            local_s = i*GROUP_SYSTEM_SIZE + j*N_SEIIR_COMPARTMENTS
            infectious_wild = infectious_wild + y[local_s + LOCAL_I1] + y[local_s + LOCAL_I2]

        for j in range(N_VARIANT_TYPE_SEIIR_GROUPS):
            local_s = i*GROUP_SYSTEM_SIZE + (j + N_WILD_TYPE_SEIIR_GROUPS)*N_SEIIR_COMPARTMENTS
            infectious_variant = infectious_variant + y[local_s + LOCAL_I1] + y[local_s + LOCAL_I2]

    infectious = np.array([infectious_wild, infectious_variant])

    # Do the thing!
    dy = np.zeros_like(y)
    for i in range(N_GROUPS):
        group_start = i * GROUP_SYSTEM_SIZE
        group_end = (i + 1) * GROUP_SYSTEM_SIZE
        group_vaccine_start = i * N_VACCINE_CATEGORIES
        group_vaccine_end = (i + 1) * N_VACCINE_CATEGORIES

        dy[group_start:group_end] = single_group_system(
            t,
            y[group_start:group_end],
            params,
            vaccines[group_vaccine_start:group_vaccine_end],
            n_total,
            infectious
        )

    return dy


@numba.njit
def variant_natural_single_group_system(t: float, y: np.ndarray, params: np.ndarray,
                                        vaccines: np.ndarray, n_total: float, infectious: np.ndarray):
    # Methodology keeps changing.  This doesn't make sense now, but might soon.
    raise NotImplementedError
    # alpha, beta, beta_b117, beta_b1351, sigma, gamma1, gamma2 = 0, 1, 2, 3, 4, 5, 6
    # b117_prevalence, b1351_prevalence = 9, 10
    #
    # infectious_wild, infectious_variant = infectious
    # b_wild = params[beta] * infectious_wild ** params[alpha] / n_total
    # b_variant_s = params[beta_variant] * infectious_variant ** params[alpha] / n_total
    # b_variant_s_variant = b_variant_s
    # return variant_single_group_system(t, y, params, b_wild, b_variant_s, b_variant_s_variant, vaccines)


@numba.njit
def variant_implicit_single_group_system(t: float, y: np.ndarray, params: np.ndarray,
                                         vaccines: np.ndarray, n_total: float, infectious: np.ndarray):
    infectious_wild, infectious_variant = infectious
    i_total = infectious.sum()

    b_wild = params[beta_wild] * params[p_wild] * i_total ** params[alpha] / n_total
    b_variant_s = params[beta_variant] * params[p_variant] * i_total ** params[alpha] / n_total
    b_variant_s_variant = params[beta_variant] * infectious_variant ** params[alpha] / n_total

    return variant_single_group_system(t, y, params, b_wild, b_variant_s, b_variant_s_variant, vaccines)


@numba.njit
def variant_explicit_single_group_system(t: float, y: np.ndarray, params: np.ndarray,
                                         vaccines: np.ndarray, n_total: float, infectious: np.ndarray):
    infectious_wild, infectious_variant = infectious

    b_combined = (params[beta_wild] * infectious_wild ** params[alpha] / n_total
                  + params[beta_variant] * infectious_variant ** params[alpha] / n_total)
    b_wild = params[p_wild] * b_combined
    b_variant_s = params[p_variant] * b_combined
    b_variant_s_variant = beta_variant * infectious_variant ** params[alpha] / n_total

    return variant_single_group_system(t, y, params, b_wild, b_variant_s, b_variant_s_variant, vaccines)


@numba.njit
def variant_single_group_system(t: float,
                                y: np.ndarray, params: np.ndarray,
                                b_wild: float, b_variant_s: float, b_variant_s_variant: float,
                                vaccines: np.ndarray):
    # Allocate our working space.  Transition matrix map.
    # Each row is a FROM compartment and each column is a TO compartment
    outflow_map = np.zeros((y.size, y.size))

    # Vaccinations from each compartment indexed by out compartment and
    # vaccine category (u, p, pa, m, ma)
    vaccines_out = get_vaccines_out(y, vaccines, params, b_wild, b_variant_s, b_variant_s_variant)

    # Unvaccinated
    # Epi transitions
    outflow_map = seiir_transition_wild(
        y,
        s, e, i1, i2, r,
        s_variant, e_variant,
        params, b_wild, b_variant_s,
        outflow_map,
    )
    # Vaccines
    # S is complicated
    outflow_map[s, s_u] += vaccines_out[s, u]
    outflow_map[s, s_p] += vaccines_out[s, p]
    outflow_map[s, s_pa] += vaccines_out[s, pa]
    outflow_map[s, s_m] += vaccines_out[s, m]
    outflow_map[s, r_ma] += vaccines_out[s, ma]

    # Other compartments are simple.
    outflow_map[e, e_u] += vaccines_out[e, u]
    outflow_map[i1, i1_u] += vaccines_out[i1, u]
    outflow_map[i2, i2_u] += vaccines_out[i2, u]
    outflow_map[r, r_u] += vaccines_out[r, u]

    # Unprotected
    outflow_map = seiir_transition_wild(
        y,
        s_u, e_u, i1_u, i2_u, r_u,
        s_u_variant, e_u_variant,
        params, b_wild, b_variant_s,
        outflow_map,
    )
    # Protected from wild-type
    outflow_map = seiir_transition_wild(
        y,
        s_p, e_p, i1_p, i2_p, r_p,
        s_u_variant, e_u_variant,
        params, b_wild, b_variant_s,
        outflow_map,
    )

    # Protected from all-types
    outflow_map = seiir_transition_wild(
        y,
        s_pa, e_pa, i1_pa, i2_pa, r_pa,
        s_pa_variant, e_pa_variant,
        params, b_wild, b_variant_s,
        outflow_map,
    )

    # Unvaccinated variant
    # Epi transitions
    outflow_map = seiir_transition(
        y,
        s_variant, e_variant, i1_variant, i2_variant, r_variant,
        params, b_variant_s_variant,
        outflow_map,
    )
    # Vaccinations
    # S is complicated
    outflow_map[s_variant, s_u_variant] += vaccines_out[s_variant, u]
    outflow_map[s_variant, s_pa_variant] += vaccines_out[s_variant, pa]
    outflow_map[s_variant, r_ma] += vaccines_out[s_variant, ma]

    # Other compartments are simple
    outflow_map[e_variant, e_u_variant] += vaccines_out[e_variant, u]
    outflow_map[i1_variant, i1_u_variant] += vaccines_out[i1_variant, u]
    outflow_map[i2_variant, i2_u_variant] += vaccines_out[i2_u_variant, u]
    outflow_map[r_variant, r_u_variant] += vaccines_out[r_variant, u]

    # Unprotected variant
    outflow_map = seiir_transition(
        y,
        s_u_variant, e_u_variant, i1_u_variant, i2_u_variant, r_u_variant,
        params, b_variant_s_variant,
        outflow_map,
    )

    # Protected variant
    outflow_map = seiir_transition(
        y,
        s_pa_variant, e_pa_variant, i1_pa_variant, i2_pa_variant, r_pa_variant,
        params, b_variant_s_variant,
        outflow_map,
    )

    # Immunized
    outflow_map[s_m, e_pa_variant] = b_variant_s_variant * y[s_m]

    inflow = outflow_map.sum(axis=0)
    outflow = outflow_map.sum(axis=1)
    result = inflow - outflow

    if result.sum() > 1e-5:
        print('Compartment mismatch: ', result.sum())

    result = compute_tracking_columns(result, outflow_map, vaccines_out)

    return result


@numba.njit
def seiir_transition_wild(y,
                          susceptible, exposed, infectious1, infectious2, removed,
                          susceptible_variant, exposed_variant,
                          params, b_wild, b_variant,
                          outflow_map):
    # Normal epi
    outflow_map[susceptible, exposed] += (b_wild + params[theta_plus]) * y[susceptible]
    outflow_map[exposed, infectious1] += params[sigma] * y[exposed]
    outflow_map[infectious1, infectious2] += params[gamma1] * y[infectious1]
    outflow_map[infectious2, removed] += params[p_cross_immune] * params[gamma2] * y[infectious2]
    # Theta minus
    outflow_map[exposed, removed] += params[theta_minus] * y[exposed]
    # Variant
    outflow_map[susceptible, exposed_variant] += b_variant * y[susceptible]
    outflow_map[infectious2, susceptible_variant] += (1 - params[p_cross_immune]) * params[gamma2] * y[infectious2]

    return outflow_map


@numba.njit
def seiir_transition(y,
                     susceptible, exposed, infectious1, infectious2, removed,
                     params, beta_s,
                     outflow_map):
    outflow_map[susceptible, exposed] += beta_s * y[susceptible]
    outflow_map[exposed, infectious1] += params[sigma] * y[exposed]
    outflow_map[infectious1, infectious2] += params[gamma1] * y[infectious1]
    outflow_map[infectious2, removed] += params[gamma2] * y[infectious2]

    return outflow_map


@numba.njit
def get_vaccines_out(y, vaccines, params, b_wild, b_variant_s, b_variant_s_variant):
    # Allocate our output space.
    vaccines_out = np.zeros((y.size, vaccines.size))
    v_total = vaccines.sum()
    # Don't vaccinate if no vaccines to deliver.
    if not v_total:
        return vaccines_out

    n_unvaccinated = count_unvaccinated(y)
    if n_unvaccinated:
        # S has many kinds of effective and ineffective vaccines
        vaccines_out = vaccinate_from_s(
            y, vaccines,
            params, b_wild, b_variant_s,
            n_unvaccinated, v_total,
            vaccines_out
        )
        # S_variant has many kinds of effective and ineffective vaccines
        vaccines_out = vaccinate_from_s_variant(
            y, vaccines,
            b_variant_s_variant,
            n_unvaccinated, v_total,
            vaccines_out
        )
        # Folks in E, I1, I2, R only unprotected.
        vaccines_out = get_unprotected_vaccines_from_not_s(
            y, vaccines_out, params, n_unvaccinated, v_total,
            e, i1, i2, r,
        )
        vaccines_out = get_unprotected_vaccines_from_not_s(
            y, vaccines_out, params, n_unvaccinated, v_total,
            e_variant, i1_variant, i2_variant, r_variant,
        )

    v_total_empirical = vaccines_out.sum()
#    if v_total:
#       percent_loss = abs(v_total_empirical - v_total) / v_total  * 100
#       if percent_loss > .05:
#           print('Losing more than 5% vaccines. Percent loss: ', percent_loss)

    return vaccines_out


@numba.njit
def get_unprotected_vaccines_from_not_s(y, vaccines_out, params, n_unvaccinated, v_total,
                                        exposed, infectious1, infectious2, removed):
    vaccines_out[exposed, u] = min(
        (1 - params[sigma] - params[theta_minus]) * y[exposed],
        y[exposed] / n_unvaccinated * v_total
    )
    vaccines_out[infectious1, u] = min(
        (1 - params[gamma1]) * y[infectious1],
        y[infectious1] / n_unvaccinated * v_total
    )
    vaccines_out[infectious2, u] = min(
        (1 - params[gamma2]) * y[infectious2],
        y[infectious2] / n_unvaccinated * v_total
    )
    vaccines_out[removed, u] = min(
        y[removed],
        y[removed] / n_unvaccinated * v_total
    )
    return vaccines_out


@numba.njit
def count_unvaccinated(y):
    unvaccinated_compartments = [
        s, e, i1, i2, r,
        s_variant, e_variant, i1_variant, i2_variant, r_variant
    ]
    n_unvaccinated = 0.
    for compartment in unvaccinated_compartments:
        n_unvaccinated = n_unvaccinated + y[compartment]
    return n_unvaccinated


@numba.njit
def vaccinate_from_s(y, vaccines,
                     params, b_wild, b_variant_s,
                     n_unvaccinated, v_total,
                     vaccines_out):
    # Folks in S can have all vaccine outcomes
    new_e_wild_from_s = b_wild * y[s] + params[theta_plus] * y[s]
    new_e_variant_from_s = b_variant_s * y[s]
    new_e_total_from_s = new_e_wild_from_s + new_e_variant_from_s

    expected_total_vaccines_s = y[s] / n_unvaccinated * v_total
    total_vaccines_s = min(y[s] - new_e_total_from_s, expected_total_vaccines_s)

    if expected_total_vaccines_s:
        for vaccine_type in [u, p, pa, m, ma]:
            expected_vaccines = y[s] / n_unvaccinated * vaccines[vaccine_type]
            rho = expected_vaccines / expected_total_vaccines_s
            vaccines_out[s, vaccine_type] = rho * total_vaccines_s
    return vaccines_out


@numba.njit
def vaccinate_from_s_variant(y, vaccines,
                             b_variant_s_variant,
                             n_unvaccinated, v_total,
                             vaccines_out):
    # Folks in S_variant can be protected or immunized from the variant
    new_e_variant_from_s_variant = b_variant_s_variant * y[s_variant]

    expected_total_vaccines_s_variant = y[s_variant] / n_unvaccinated * v_total
    total_vaccines_s_variant = min(y[s_variant] - new_e_variant_from_s_variant,
                                   expected_total_vaccines_s_variant)

    if expected_total_vaccines_s_variant:
        total_ineffective = vaccines[u] + vaccines[p] + vaccines[m]
        expected_u_vaccines = y[s_variant] / n_unvaccinated * total_ineffective
        rho = expected_u_vaccines / expected_total_vaccines_s_variant
        vaccines_out[s_variant, u] = rho * total_vaccines_s_variant

        for vaccine_type in [pa, ma]:
            expected_vaccines = y[s_variant] / n_unvaccinated * vaccines[vaccine_type]
            rho = expected_vaccines / expected_total_vaccines_s_variant
            vaccines_out[s_variant, vaccine_type] = rho * total_vaccines_s_variant
    return vaccines_out


@numba.njit
def compute_tracking_columns(result, outflow_map, vaccines_out):
    # New wild type infections
    result[new_e_wild] = (
        outflow_map[s, e]
        + outflow_map[s_u, e_u]
        + outflow_map[s_p, e_p]
        + outflow_map[s_pa, e_pa]
    )
    # New variant type infections
    result[new_e_variant] = (
        outflow_map[s, e_variant]
        + outflow_map[s_variant, e_variant]
        + outflow_map[s_u, e_u_variant]
        + outflow_map[s_u_variant, e_u_variant]
        + outflow_map[s_p, e_u_variant]
        + outflow_map[s_pa, e_pa_variant]
        + outflow_map[s_pa_variant, e_pa_variant]
        + outflow_map[s_m, e_pa_variant]
    )
    # New wild type protected infections
    result[new_e_p_wild] = (
        outflow_map[s_p, e_p]
        + outflow_map[s_pa, e_pa]
    )
    # New variant type protected infections
    result[new_e_p_variant] = (
        outflow_map[s_pa, e_pa_variant]
        + outflow_map[s_pa_variant, e_pa_variant]
        + outflow_map[s_m, e_pa_variant]
    )

    # New variant type infections breaking through natural immunity
    result[new_e_nbt] = (
        outflow_map[s_variant, e_variant]
        + outflow_map[s_u_variant, e_u_variant]
        + outflow_map[s_pa_variant, e_pa_variant]
    )
    # New variant type infections breaking through vaccine immunity
    result[new_e_vbt] = outflow_map[s_m, e_pa_variant]

    # Proportion cross immune checks
    result[new_s_v] = (
        outflow_map[i2, s_variant]
        + outflow_map[i2_u,  s_u_variant]
        + outflow_map[i2_p, s_u_variant]
        + outflow_map[i2_pa, s_pa_variant]
    )

    result[new_r_w] = (
        outflow_map[i2, r]
        + outflow_map[i2_u, r_u]
        + outflow_map[i2_p, r_p]
        + outflow_map[i2_pa, r_pa]
    )

    result[v_u] = vaccines_out[:, u].sum()
    result[v_p] = vaccines_out[:, p].sum()
    result[v_m] = vaccines_out[:, m].sum()
    result[v_pa] = vaccines_out[:, pa].sum()
    result[v_ma] = vaccines_out[:, ma].sum()

    return result
