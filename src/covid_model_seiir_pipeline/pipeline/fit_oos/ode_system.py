from collections import namedtuple

import numba
import numpy as np


##############################################
# Give indices semantically meaningful names #
##############################################
# Effectively we're making a bunch of Enums but with nicer properties for
# interacting with dataframes outside this module.
Parameters = namedtuple(
    'Parameters', [
        'alpha', 'sigma', 'gamma1', 'gamma2', 'new_e',
        'kappa', 'rho', 'phi', 'pi', 'epsilon', 'rho_variant', 'a', 'b',
        'p_cross_immune',
    ]
)
parameters = Parameters(*list(range(len(Parameters._fields))))

VaccineTypes = namedtuple(
    'Vaccines', [
        'u', 'p', 'pa', 'm', 'ma',
    ]
)
vaccine_types = VaccineTypes(*list(range(len(VaccineTypes._fields))))

Compartments = namedtuple(
    'Compartments', [
        'S',            'E',            'I1',            'I2',            'R',
        'S_u',          'E_u',          'I1_u',          'I2_u',          'R_u',
        'S_p',          'E_p',          'I1_p',          'I2_p',          'R_p',
        'S_pa',         'E_pa',         'I1_pa',         'I2_pa',         'R_pa',

        'S_variant',    'E_variant',    'I1_variant',    'I2_variant',    'R_variant',
        'S_variant_u',  'E_variant_u',  'I1_variant_u',  'I2_variant_u',  'R_variant_u',
        'S_variant_pa', 'E_variant_pa', 'I1_variant_pa', 'I2_variant_pa', 'R_variant_pa',

        'S_m',                                                            'R_m',
    ]
)
compartments = Compartments(*list(range(len(Compartments._fields))))

TrackingCompartments = namedtuple(
    'TrackingCompartments', [
        'NewE_wild', 'NewE_variant', 'NewE_p_wild', 'NewE_p_variant',
        'NewE_nbt', 'NewE_vbt', 'NewS_v', 'NewR_w',
        'V_u', 'V_p', 'V_pa', 'V_m', 'V_ma',
    ]
)
tracking_compartments = TrackingCompartments(*[i + len(Compartments._fields)
                                               for i in range(len(TrackingCompartments._fields))])

_UNVACCINATED = np.array([
    compartments.S,  compartments.S_variant,
    compartments.E,  compartments.E_variant,
    compartments.I1, compartments.I1_variant,
    compartments.I2, compartments.I2_variant,
    compartments.R,  compartments.R_variant,
])
_SUSCEPTIBLE_WILD = np.array([
    compartments.S, compartments.S_u, compartments.S_p, compartments.S_pa,
])
_SUSCEPTIBLE_VARIANT_ONLY = np.array([
    compartments.S_variant, compartments.S_variant_u, compartments.S_variant_pa, compartments.S_m,
])
_INFECTIOUS_WILD = np.array([
    compartments.I1,    compartments.I2,
    compartments.I1_u,  compartments.I2_u,
    compartments.I1_p,  compartments.I2_p,
    compartments.I1_pa, compartments.I2_pa,
])
_INFECTIOUS_VARIANT = np.array([
    compartments.I1_variant,    compartments.I2_variant,
    compartments.I1_variant_u,  compartments.I2_variant_u,
    compartments.I2_variant_pa, compartments.I2_variant_pa,
])


@numba.njit
def single_force_system(t: float, y: np.ndarray, params: np.ndarray):
    alpha, pi, epsilon, rho_variant = params[np.array([
        parameters.alpha, parameters.pi, parameters.epsilon, parameters.rho_variant
    ])]
    i_variant = y[_INFECTIOUS_VARIANT].sum()
    if rho_variant and not i_variant:
        y = delta_shift(
            y, alpha, pi, epsilon,
            compartments.S, compartments.E_variant, compartments.I1_variant,
        )
        y = delta_shift(
            y, alpha, pi, epsilon,
            compartments.S_u, compartments.E_variant_u, compartments.I1_variant_u,
        )
        y = delta_shift(
            y, alpha, pi, epsilon,
            compartments.S_p, compartments.E_variant_u, compartments.I1_variant_u,
        )
        y = delta_shift(
            y, alpha, pi, epsilon,
            compartments.S_pa, compartments.E_variant_pa, compartments.I1_variant_pa,
        )

    infectious_wild = y[_INFECTIOUS_WILD].sum()
    infectious_variant = y[_INFECTIOUS_VARIANT].sum()

    return system(t, y, params, infectious_wild, infectious_variant)


@numba.njit
def ramp_force_system(t: float, y: np.ndarray, params: np.ndarray):
    alpha, rho_variant, a, b = params[np.array([
        parameters.alpha, parameters.rho_variant, parameters.a, parameters.b,
    ])]

    infectious_wild = y[_INFECTIOUS_WILD].sum()
    infectious_variant = y[_INFECTIOUS_VARIANT].sum()
    infectious_total = infectious_wild + infectious_variant

    if rho_variant < a:
        lower_bound = infectious_total * rho_variant
    elif rho_variant < b:
        z = infectious_total * rho_variant
        lower_bound = z + (rho_variant - a) / (b - a) * (infectious_variant - z)
    else:
        lower_bound = infectious_variant
    infectious_variant = max(lower_bound, infectious_variant)
    infectious_wild = infectious_total - infectious_variant

    return system(t, y, params, infectious_wild, infectious_variant)


@numba.njit
def delta_shift(y,
                alpha, pi, epsilon,
                susceptible, exposed, infectious1):
    delta = min(max(pi * y[exposed], epsilon), 1/2 * y[susceptible])
    y[susceptible] -= delta + (delta / 5)**(1 / alpha)
    y[exposed] += delta
    y[infectious1] += (delta / 5)**(1 / alpha)
    return y


@numba.njit
def system(t: float, y: np.ndarray, params: np.ndarray, infectious_wild: float, infectious_variant: float):
    params, vaccines = params[:len(parameters)], params[len(parameters):]

    outflow_map = np.zeros((y.size, y.size))

    susceptible_wild = y[_SUSCEPTIBLE_WILD].sum()
    susceptible_variant_only = y[_SUSCEPTIBLE_VARIANT_ONLY].sum()
    new_e_wild, new_e_variant_naive, new_e_variant_reinf = split_new_e(y, params, infectious_wild, infectious_variant)
    vaccines_out = get_vaccines_out(
        y,
        params, vaccines,
        new_e_wild, new_e_variant_naive, new_e_variant_reinf
    )
    # Unvaccinated
    # Epi transitions
    outflow_map = seiir_transition_wild(
        y, params,
        new_e_wild * y[compartments.S] / susceptible_wild,
        new_e_variant_naive * y[compartments.S] / susceptible_wild,
        compartments.S, compartments.E, compartments.I1, compartments.I2, compartments.R,
        compartments.S_variant, compartments.E_variant,
        outflow_map,
    )
    # Vaccines
    # S is complicated
    outflow_map[compartments.S, compartments.S_u] += vaccines_out[compartments.S, vaccine_types.u]
    outflow_map[compartments.S, compartments.S_p] += vaccines_out[compartments.S, vaccine_types.p]
    outflow_map[compartments.S, compartments.S_pa] += vaccines_out[compartments.S, vaccine_types.pa]
    outflow_map[compartments.S, compartments.S_m] += vaccines_out[compartments.S, vaccine_types.m]
    outflow_map[compartments.S, compartments.R_m] += vaccines_out[compartments.S, vaccine_types.ma]

    # Other compartments are simple.
    outflow_map[compartments.E, compartments.E_u] += vaccines_out[compartments.E, vaccine_types.u]
    outflow_map[compartments.I1, compartments.I1_u] += vaccines_out[compartments.I1, vaccine_types.u]
    outflow_map[compartments.I2, compartments.I2_u] += vaccines_out[compartments.I2, vaccine_types.u]
    outflow_map[compartments.R, compartments.R_u] += vaccines_out[compartments.R, vaccine_types.u]

    # Unprotected
    outflow_map = seiir_transition_wild(
        y, params,
        safe_divide(new_e_wild * y[compartments.S_u],  susceptible_wild),
        safe_divide(new_e_variant_naive * y[compartments.S_u], susceptible_wild),
        compartments.S_u, compartments.E_u, compartments.I1_u, compartments.I2_u, compartments.R_u,
        compartments.S_variant_u, compartments.E_variant_u,
        outflow_map,
    )

    # Protected from wild-type
    outflow_map = seiir_transition_wild(
        y, params,
        safe_divide(new_e_wild * y[compartments.S_p], susceptible_wild),
        safe_divide(new_e_variant_naive * y[compartments.S_p], susceptible_wild),
        compartments.S_p, compartments.E_p, compartments.I1_p, compartments.I2_p, compartments.R_p,
        compartments.S_variant_u, compartments.E_variant_u,
        outflow_map,
    )

    # Protected from all types
    outflow_map = seiir_transition_wild(
        y, params,
        safe_divide(new_e_wild * y[compartments.S_pa], susceptible_wild),
        safe_divide(new_e_variant_naive * y[compartments.S_pa], susceptible_wild),
        compartments.S_pa, compartments.E_pa, compartments.I1_pa, compartments.I2_pa, compartments.R_pa,
        compartments.S_variant_pa, compartments.E_variant_pa,
        outflow_map,
    )

    # Unvaccinated variant
    # Epi transitions
    outflow_map = seiir_transition_variant(
        y, params,
        safe_divide(new_e_variant_reinf * y[compartments.S_variant], susceptible_variant_only),
        compartments.S_variant, compartments.E_variant, compartments.I1_variant,
        compartments.I2_variant, compartments.R_variant,
        outflow_map,
    )

    # Vaccinations
    # S is complicated
    outflow_map[compartments.S_variant, compartments.S_variant_u] += vaccines_out[
        compartments.S_variant, vaccine_types.u]
    outflow_map[compartments.S_variant, compartments.S_variant_pa] += vaccines_out[
        compartments.S_variant, vaccine_types.pa]
    outflow_map[compartments.S_variant, compartments.R_m] += vaccines_out[
        compartments.S_variant, vaccine_types.ma]

    # Other compartments are simple
    outflow_map[compartments.E_variant, compartments.E_variant_u] += vaccines_out[
        compartments.E_variant, vaccine_types.u]
    outflow_map[compartments.I1_variant, compartments.I1_variant_u] += vaccines_out[
        compartments.I1_variant, vaccine_types.u]
    outflow_map[compartments.I2_variant, compartments.I2_variant_u] += vaccines_out[
        compartments.I2_variant, vaccine_types.u]
    outflow_map[compartments.R_variant, compartments.R_variant_u] += vaccines_out[
        compartments.R_variant, vaccine_types.u]

    # Unprotected variant
    outflow_map = seiir_transition_variant(
        y, params,
        safe_divide(new_e_variant_reinf * y[compartments.S_variant_u], susceptible_variant_only),
        compartments.S_variant_u, compartments.E_variant_u, compartments.I1_variant_u,
        compartments.I2_variant_u, compartments.R_variant_u,
        outflow_map,
    )

    # Protected variant
    outflow_map = seiir_transition_variant(
        y, params,
        safe_divide(new_e_variant_reinf * y[compartments.S_variant_pa], susceptible_variant_only),
        compartments.S_variant_pa, compartments.E_variant_pa, compartments.I1_variant_pa,
        compartments.I2_variant_pa, compartments.R_variant_pa,
        outflow_map,
    )

    # Immunized
    outflow_map[compartments.S_m, compartments.E_variant_pa] = (
        safe_divide(new_e_variant_reinf * y[compartments.S_m], susceptible_variant_only)
    )

    inflow = outflow_map.sum(axis=0)
    outflow = outflow_map.sum(axis=1)
    result = inflow - outflow

    if result.sum() > 1e-5:
        print('Compartment mismatch: ', result.sum())

    result = compute_tracking_columns(result, outflow_map, vaccines_out)

    return result


@numba.njit
def split_new_e(y, params, infectious_wild, infectious_variant):
    susceptible_wild = y[_SUSCEPTIBLE_WILD].sum()
    susceptible_variant_only = y[_SUSCEPTIBLE_VARIANT_ONLY].sum()

    alpha, kappa, rho, phi = params[np.array([parameters.alpha, parameters.kappa, parameters.rho, parameters.phi])]
    new_e = params[parameters.new_e]

    scale_wild = (1 + kappa * rho)
    scale_variant = (1 + kappa * phi)
    scale = scale_variant / scale_wild

    si_wild = susceptible_wild * infectious_wild**alpha
    si_variant_naive = scale * susceptible_wild * infectious_variant**alpha
    si_variant_reinf = scale * susceptible_variant_only * infectious_variant**alpha

    z = si_wild + scale * (si_variant_naive + si_variant_reinf)

    new_e_wild = si_wild / z * new_e
    new_e_variant_naive = si_variant_naive / z * new_e
    new_e_variant_reinf = si_variant_reinf / z * new_e
    return np.array([new_e_wild, new_e_variant_naive, new_e_variant_reinf])


@numba.njit
def get_vaccines_out(y,
                     params, vaccines,
                     new_exposed_wild, new_exposed_variant_naive, new_exposed_variant_reinf):
    # Allocate our output space.
    vaccines_out = np.zeros((y.size, len(vaccines)))

    v_total = vaccines.sum()
    n_unvaccinated = y[_UNVACCINATED].sum()

    # Don't vaccinate if no vaccines to deliver.
    if not v_total or not n_unvaccinated:
        return vaccines_out

    # S has many kinds of effective and ineffective vaccines
    vaccines_out = vaccinate_from_s(
        y,
        vaccines,
        new_exposed_wild + new_exposed_variant_naive,
        v_total, n_unvaccinated,
        vaccines_out,
    )
    # S_variant has many kinds of effective and ineffective vaccines
    vaccines_out = vaccinate_from_s_variant(
        y,
        vaccines,
        new_exposed_variant_reinf,
        v_total, n_unvaccinated,
        vaccines_out,
    )
    # Folks in E, I1, I2, R only unprotected.
    vaccines_out = get_unprotected_vaccines_from_not_s(
        y, params,
        v_total, n_unvaccinated,
        compartments.E, compartments.I1, compartments.I2, compartments.R,
        vaccines_out,
    )
    vaccines_out = get_unprotected_vaccines_from_not_s(
        y, params,
        v_total, n_unvaccinated,
        compartments.E_variant, compartments.I1_variant, compartments.I2_variant, compartments.R_variant,
        vaccines_out,
    )

    return vaccines_out


@numba.njit
def vaccinate_from_s(y,
                     vaccines,
                     new_e_from_s,
                     v_total, n_unvaccinated,
                     vaccines_out):
    expected_total_vaccines_s = y[compartments.S] / n_unvaccinated * v_total
    total_vaccines_s = min(y[compartments.S] - new_e_from_s, expected_total_vaccines_s)

    if expected_total_vaccines_s:
        for vaccine_type in vaccine_types:
            expected_vaccines = y[compartments.S] / n_unvaccinated * vaccines[vaccine_type]
            vaccine_ratio = expected_vaccines / expected_total_vaccines_s
            vaccines_out[compartments.S, vaccine_type] = vaccine_ratio * total_vaccines_s
    return vaccines_out


@numba.njit
def vaccinate_from_s_variant(y,
                             vaccines,
                             new_e_from_s_variant,
                             v_total, n_unvaccinated,
                             vaccines_out):
    expected_total_vaccines_s_variant = y[compartments.S_variant] / n_unvaccinated * v_total
    total_vaccines_s_variant = min(y[compartments.S_variant] - new_e_from_s_variant, expected_total_vaccines_s_variant)

    if expected_total_vaccines_s_variant:
        total_ineffective = vaccines[vaccine_types.u] + vaccines[vaccine_types.p] + vaccines[vaccine_types.m]
        expected_u_vaccines = y[compartments.S_variant] / n_unvaccinated * total_ineffective
        vaccine_ratio = expected_u_vaccines / expected_total_vaccines_s_variant
        vaccines_out[compartments.S_variant, vaccine_types.u] = vaccine_ratio * total_vaccines_s_variant

        for vaccine_type in [vaccine_types.pa, vaccine_types.ma]:
            expected_vaccines = y[compartments.S_variant] / n_unvaccinated * vaccines[vaccine_type]
            vaccine_ratio = expected_vaccines / expected_total_vaccines_s_variant
            vaccines_out[compartments.S_variant, vaccine_type] = vaccine_ratio * total_vaccines_s_variant

    return vaccines_out


@numba.njit
def get_unprotected_vaccines_from_not_s(y,
                                        params,
                                        v_total, n_unvaccinated,
                                        exposed, infectious1, infectious2, removed,
                                        vaccines_out):
    param_map = (
        (exposed, params[parameters.sigma]),
        (infectious1, params[parameters.gamma1]),
        (infectious2, params[parameters.gamma2]),
    )

    for compartment, param in param_map:
        vaccines_out[compartment, vaccine_types.u] = min(
            1 - param * y[compartment],
            y[compartment] / n_unvaccinated * v_total
        )
    vaccines_out[removed, vaccine_types.u] = min(
        y[removed],
        y[removed] / n_unvaccinated * v_total
    )

    return vaccines_out


@numba.njit
def seiir_transition_wild(y, params,
                          new_e_wild, new_e_variant,
                          susceptible, exposed, infectious1, infectious2, removed,
                          susceptible_variant, exposed_variant,
                          outflow_map):
    outflow_map[susceptible, exposed] += new_e_wild
    outflow_map[exposed, infectious1] += params[parameters.sigma] * y[exposed]
    outflow_map[infectious1, infectious2] += params[parameters.gamma1] * y[infectious1]
    outflow_map[infectious2, removed] += params[parameters.p_cross_immune] * params[parameters.gamma2] * y[infectious2]

    outflow_map[susceptible, exposed_variant] += new_e_variant
    outflow_map[infectious2, susceptible_variant] += (
        (1 - params[parameters.p_cross_immune]) * params[parameters.gamma2] * y[infectious2]
    )
    return outflow_map


@numba.njit
def seiir_transition_variant(y, params,
                             new_e,
                             susceptible, exposed, infectious1, infectious2, removed,
                             outflow_map):
    import pdb; pdb.set_trace()
    outflow_map[susceptible, exposed] += new_e
    outflow_map[exposed, infectious1] += params[parameters.sigma] * y[exposed]
    outflow_map[infectious1, infectious2] += params[parameters.gamma1] * y[infectious1]
    outflow_map[infectious2, removed] += params[parameters.gamma2] * y[infectious2]

    return outflow_map


@numba.njit
def compute_tracking_columns(result, outflow_map, vaccines_out):
    # New wild type infections
    result[tracking_compartments.NewE_wild] = (
        outflow_map[compartments.S, compartments.E]
        + outflow_map[compartments.S_u, compartments.E_u]
        + outflow_map[compartments.S_p, compartments.E_p]
        + outflow_map[compartments.S_pa, compartments.E_pa]
    )
    # New variant type infections
    result[tracking_compartments.NewE_variant] = (
        outflow_map[compartments.S, compartments.E_variant]
        + outflow_map[compartments.S_variant, compartments.E_variant]
        + outflow_map[compartments.S_u, compartments.E_variant_u]
        + outflow_map[compartments.S_variant_u, compartments.E_variant_u]
        + outflow_map[compartments.S_p, compartments.E_variant_u]
        + outflow_map[compartments.S_pa, compartments.E_variant_pa]
        + outflow_map[compartments.S_variant_pa, compartments.E_variant_pa]
        + outflow_map[compartments.S_m, compartments.E_variant_pa]
    )
    # New wild type protected infections
    result[tracking_compartments.NewE_p_wild] = (
        outflow_map[compartments.S_p, compartments.E_p]
        + outflow_map[compartments.S_pa, compartments.E_pa]
    )
    # New variant type protected infections
    result[tracking_compartments.NewE_p_variant] = (
        outflow_map[compartments.S_pa, compartments.E_variant_pa]
        + outflow_map[compartments.S_variant_pa, compartments.E_variant_pa]
        + outflow_map[compartments.S_m, compartments.E_variant_pa]
    )

    # New variant type infections breaking through natural immunity
    result[tracking_compartments.NewE_nbt] = (
        outflow_map[compartments.S_variant, compartments.E_variant]
        + outflow_map[compartments.S_variant_u, compartments.E_variant_u]
        + outflow_map[compartments.S_variant_pa, compartments.E_variant_pa]
    )
    # New variant type infections breaking through vaccine immunity
    result[tracking_compartments.NewE_vbt] = outflow_map[compartments.S_m, compartments.E_variant_pa]

    # Proportion cross immune checks
    result[tracking_compartments.NewS_v] = (
        outflow_map[compartments.I2, compartments.S_variant]
        + outflow_map[compartments.I2_u,  compartments.S_variant_u]
        + outflow_map[compartments.I2_p, compartments.S_variant_u]
        + outflow_map[compartments.I2_pa, compartments.S_variant_pa]
    )

    result[tracking_compartments.NewR_w] = (
        outflow_map[compartments.I2, compartments.R]
        + outflow_map[compartments.I2_u,  compartments.R_u]
        + outflow_map[compartments.I2_p, compartments.S_u]
        + outflow_map[compartments.I2_pa, compartments.S_u]
    )

    result[tracking_compartments.V_u] = vaccines_out[:, vaccine_types.u].sum()
    result[tracking_compartments.V_p] = vaccines_out[:, vaccine_types.p].sum()
    result[tracking_compartments.V_m] = vaccines_out[:, vaccine_types.m].sum()
    result[tracking_compartments.V_pa] = vaccines_out[:, vaccine_types.pa].sum()
    result[tracking_compartments.V_ma] = vaccines_out[:, vaccine_types.ma].sum()

    return result


@numba.njit
def safe_divide(a: float, b: float):
    if b == 0.0:
        assert a == 0.0
        return 0.0
    return a / b
