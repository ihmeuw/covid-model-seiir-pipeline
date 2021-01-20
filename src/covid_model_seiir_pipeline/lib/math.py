import numpy as np
import pandas as pd


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
    covariates = covariates.set_index(['location_id', 'date']).sort_index()
    covariates['intercept'] = 1.0
    coefficients = coefficients.set_index(['location_id']).sort_index()
    return (covariates * coefficients).sum(axis=1)


def get_observed_infecs_and_deaths(infection_data: pd.DataFrame):
    """Gets observed data out of infectionator outputs."""
    observed = infection_data['observed_infections'] == 1
    observed_infections = (infection_data
                           .loc[observed, ['location_id', 'date', 'infections_draw']]
                           .set_index(['location_id', 'date'])
                           .sort_index()
                           .rename(columns={'infections_draw': 'infections'}))
    observed = infection_data['observed_deaths'] == 1
    observed_deaths = (infection_data
                       .loc[observed, ['location_id', 'date', 'deaths']]
                       .set_index(['location_id', 'date'])
                       .sort_index())
    observed_deaths['observed'] = 1
    return observed_infections, observed_deaths


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
