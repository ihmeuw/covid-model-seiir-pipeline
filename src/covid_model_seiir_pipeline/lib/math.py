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
