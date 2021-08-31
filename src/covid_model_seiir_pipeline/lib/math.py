import numba
import numpy as np
import pandas as pd
import scipy.stats


SOLVER_DT = 0.1


def compute_beta_hat(covariates: pd.DataFrame, coefficients: pd.DataFrame) -> pd.Series:
    """Computes beta from a set of covariates and their coefficients.

    We're leveraging regression coefficients and past or future values for
    covariates to produce a modelled beta (beta hat). Past data is used
    in the original regression to produce the coefficients so that beta hat
    best matches the data.

    .. math::

        \hat{\beta}(location, time) = \sum\limits_{c \in cov} coeff_c(location) * covariate_c(location, time)

    Parameters
    ----------
    covariates
        DataFrame with columns 'location_id', 'date', and a column for
        each covariate. A time series for the covariate values by location.
    coefficients
        DataFrame with a 'location_id' column and a column for each covariate
        representing the strength of the relationship between the covariate
        and beta.

    """
    covariates['intercept'] = 1.0
    return (covariates * coefficients).sum(axis=1)


def adjust_vaccinations(vaccine_data: pd.DataFrame):
    risk_groups = ['lr', 'hr']
    vaccinations = {}

    for risk_group in risk_groups:
        base_col_map = {
            f'unprotected_{risk_group}': f'unprotected_{risk_group}',
            f'protected_wild_type_{risk_group}': f'effective_protected_wildtype_{risk_group}',
            f'protected_all_types_{risk_group}': f'effective_protected_variant_{risk_group}',
            f'immune_wild_type_{risk_group}': f'effective_wildtype_{risk_group}',
            f'immune_all_types_{risk_group}': f'effective_variant_{risk_group}',
        }
        for to_name, from_name in base_col_map.items():
            vaccinations[to_name] = vaccine_data[from_name].rename(to_name)
    return vaccinations


def solve_ode(system, t, init_cond, params):
    t_solve = np.arange(np.min(t), np.max(t) + SOLVER_DT, SOLVER_DT / 2)
    y_solve = np.zeros((init_cond.size, t_solve.size),
                       dtype=init_cond.dtype)
    y_solve[:, 0] = init_cond
    # linear interpolate the parameters
    params = linear_interpolate(t_solve, t, params)
    y_solve = _rk45(system, t_solve, y_solve, params, SOLVER_DT)
    # linear interpolate the solutions.
    y_solve = linear_interpolate(t, t_solve, y_solve)
    return y_solve


@numba.njit
def _rk45(system,
          t_solve: np.array,
          y_solve: np.array,
          params: np.array,
          dt: float):
    for i in range(2, t_solve.size, 2):
        k1 = system(t_solve[i - 2], y_solve[:, i - 2], params[:, i - 2])
        k2 = system(t_solve[i - 1], y_solve[:, i - 2] + dt / 2 * k1, params[:, i - 1])
        k3 = system(t_solve[i - 1], y_solve[:, i - 2] + dt / 2 * k2, params[:, i - 1])
        k4 = system(t_solve[i], y_solve[:, i - 2] + dt * k3, params[:, i])
        y_solve[:, i] = y_solve[:, i - 2] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_solve


@numba.njit
def safe_divide(a: float, b: float):
    """Divide that returns zero if numerator and denominator are both zero."""
    if b == 0.0:
        assert a == 0.0
        return 0.0
    return a / b


def linear_interpolate(t_target: np.ndarray,
                       t_org: np.ndarray,
                       x_org: np.ndarray) -> np.ndarray:
    is_vector = x_org.ndim == 1
    if is_vector:
        x_org = x_org[None, :]

    assert t_org.size == x_org.shape[1]

    x_target = np.vstack([
        np.interp(t_target, t_org, x_org[i])
        for i in range(x_org.shape[0])
    ])

    if is_vector:
        return x_target.ravel()
    else:
        return x_target


def get_waning_dist(ode_parameters):
    start = ode_parameters.waning_start.mean()
    mean = ode_parameters.waning_mean.mean()
    var = ode_parameters.waning_sd.mean()**2
    return scipy.stats.gamma(loc=start, scale=var/mean, a=mean**2/var)


def sample_dist(dist, t, rate=1):
    dist_cdf = dist.cdf(t[::rate])
    dist_discrete_pdf = dist_cdf[1:] - dist_cdf[:-1]
    dist_discrete_pdf = np.hstack([dist_discrete_pdf, dist_discrete_pdf[-1]])
    sampled_dist = np.zeros_like(t, dtype=dist_cdf.dtype)
    sampled_dist[::rate] = dist_discrete_pdf
    return sampled_dist
