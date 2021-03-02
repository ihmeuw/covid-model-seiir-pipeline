import numba
import numpy as np

##############################################
# Give indices semantically meaningful names #
##############################################
PARAMETERS = [
    'alpha', 'beta', 'sigma', 'gamma1', 'gamma2',
    'theta_plus', 'theta_minus'
]
(
    alpha, beta, sigma, gamma1, gamma2,
    theta_plus, theta_minus
) = range(len(PARAMETERS))

COMPARTMENTS = (
    'S',    'E',    'I1',    'I2',    'R',
    'S_u',  'E_u',  'I1_u',  'I2_u',  'R_u',
    'S_p',  'E_p',  'I1_p',  'I2_p',  'R_p',
    'S_pa', 'E_pa', 'I1_pa', 'I2_pa', 'R_pa',
    'S_m'                             'R_m',
)
(
    s,    e,    i1,    i2,    r,
    s_u,  e_u,  i1_u,  i2_u,  r_u,
    s_p,  e_p,  i1_p,  i2_p,  r_p,
    s_pa, e_pa, i1_pa, i2_pa, r_pa,
    s_m,                      r_m,
) = range(len(COMPARTMENTS))

VACCINE_CATEGORIES = (
    'u', 'p', 'pa', 'm', 'ma',
)
(
    u, p, pa, m, ma,
) = range(len(VACCINE_CATEGORIES))


N_SEIIR_COMPARTMENTS = 5

# 3rd and 4th compartment of each seiir group are infectious.
LOCAL_I1 = 2
LOCAL_I2 = 3


@numba.njit
def system(t: float, y: np.ndarray, params: np.array):
    system_size = len(COMPARTMENTS)
    n_groups = y.size // system_size
    n_vaccines = len(VACCINE_CATEGORIES) * n_groups

    # Split parameters from vaccines.
    params, vaccines = params[:-n_vaccines], params[-n_vaccines:]

    # Demographic groups mix, so we need to precompute infectious folks.
    infectious = 0.
    n_total = y.sum()
    for i in range(n_groups):
        # len(vaccine_categories) - 1 for immune + 1 for unvaccinated
        for j in range(len(VACCINE_CATEGORIES)):
            # 3rd and 4th compartment of each group + seiir are infectious.
            local_s = i*system_size + j*N_SEIIR_COMPARTMENTS
            infectious = (infectious + y[local_s + LOCAL_I1] + y[local_s + LOCAL_I2])

    dy = np.zeros_like(y)
    for i in range(n_groups):
        group_start = i * system_size
        group_end = (i + 1) * system_size
        group_vaccine_start = i * len(VACCINE_CATEGORIES)
        group_vaccine_end = (i + 1) * len(VACCINE_CATEGORIES)

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
def single_group_system(t: float,
                        y: np.ndarray, params: np.ndarray,
                        vaccines: np.array, n_total: float, infectious: float):
    # Allocate our working space.  Transition matrix map.
    # Each row is a FROM compartment and each column is a TO compartment
    outflow_map = np.zeros((y.size, y.size))

    # b is the relative rate of infections from S compartments.
    b = params[beta] * infectious**params[alpha] / n_total

    # Vaccinations from each compartment indexed by out compartment and
    # vaccine category (u, p, ma)
    vaccines_out = get_vaccines_out(y, vaccines, params, b)

    # Unvaccinated
    # Epi transitions
    outflow_map = seiir_transition(
        y, params, b,
        s, e, i1, i2, r,
        outflow_map,
    )
    # Vaccines
    outflow_map[s, s_u] += vaccines_out[s, u]
    outflow_map[s, s_p] += vaccines_out[s, p]
    outflow_map[s, s_pa] += vaccines_out[s, pa]
    outflow_map[s, s_m] += vaccines_out[s, m]
    outflow_map[s, r_m] += vaccines_out[s, ma]

    outflow_map[e, e_u] += vaccines_out[e, u]
    outflow_map[i1, i1_u] += vaccines_out[i1, u]
    outflow_map[i2, i2_u] += vaccines_out[i2, u]
    outflow_map[r, r_u] += vaccines_out[r, u]

    # Unprotected
    outflow_map = seiir_transition(
        y, params, b,
        s_u, e_u, i1_u, i2_u, r_u,
        outflow_map,
    )

    # Protected
    outflow_map = seiir_transition(
        y, params, b,
        s_p, e_p, i1_p, i2_p, r_p,
        outflow_map,
    )

    # Protected all
    outflow_map = seiir_transition(
        y, params, b,
        s_pa, e_pa, i1_pa, i2_pa, r_pa,
        outflow_map,
    )

    inflow = outflow_map.sum(axis=0)
    outflow = outflow_map.sum(axis=1)
    result = inflow - outflow

    if result.sum() > 1e-5:
        print('Compartment mismatch: ', result.sum())

    return result


@numba.njit
def get_vaccines_out(y: np.ndarray, vaccines: np.ndarray, params: np.ndarray, b: float) -> np.ndarray:
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
            y, vaccines, params, b,
            n_unvaccinated, v_total,
            vaccines_out,
        )
        vaccines_out = vaccinate_from_not_s(
            y, params,
            n_unvaccinated, v_total,
            e, i1, i2, r,
            vaccines_out,
        )

    return vaccines_out


@numba.njit
def count_unvaccinated(y: np.ndarray) -> float:
    unvaccinated_compartments = [
        s, e, i1, i2, r,
    ]
    n_unvaccinated = 0.
    for compartment in unvaccinated_compartments:
        n_unvaccinated = n_unvaccinated + y[compartment]
    return n_unvaccinated


@numba.njit
def vaccinate_from_s(y: np.ndarray, vaccines: np.ndarray, params: np.ndarray, b: float,
                     n_unvaccinated, v_total,
                     vaccines_out):
    # Folks in S can have all vaccine outcomes
    new_e = b * y[s] + params[theta_plus] * y[s]

    expected_total_vaccines_s = y[s] / n_unvaccinated * v_total
    total_vaccines_s = min(y[s] - new_e, expected_total_vaccines_s)

    if expected_total_vaccines_s:
        for vaccine_type in [u, p, pa, m, ma]:
            expected_vaccines = y[s] / n_unvaccinated * vaccines[vaccine_type]
            rho = expected_vaccines / expected_total_vaccines_s
            vaccines_out[s, vaccine_type] = rho * total_vaccines_s
    return vaccines_out


@numba.njit
def vaccinate_from_not_s(y: np.ndarray, params: np.ndarray,
                         n_unvaccinated: float, v_total: float,
                         exposed: int, infectious1: int, infectious2: int, removed: int,
                         vaccines_out: np.ndarray):
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
def seiir_transition(y: np.ndarray, params: np.ndarray, b: float,
                     susceptible: int, exposed: int, infectious1: int, infectious2: int, removed: int,
                     outflow_map: np.ndarray):
    # Normal epi + theta plus
    outflow_map[susceptible, exposed] += (b + params[theta_plus]) * y[susceptible]
    outflow_map[exposed, infectious1] += params[sigma] * y[exposed]
    outflow_map[infectious1, infectious2] += params[gamma1] * y[infectious1]
    outflow_map[infectious2, removed] += params[gamma2] * y[infectious2]
    # Theta minus
    outflow_map[exposed, removed] += params[theta_minus] * y[exposed]

    return outflow_map
