import pandas as pd

from covid_model_seiir_pipeline.forecasting.model import ODERunner


class ModelRunner:

    @staticmethod
    def forecast(model_specs, init_cond, times, betas, thetas=None, dt=0.1):
        """
        Solves ode for given time and beta

        Arguments:
            model_specs (SeiirModelSpecs): specification for the model. See
                covid_model_seiir_pipeline.forecasting.model.SeiirModelSpecs
                for more details.
                example:
                    model_specs = SeiirModelSpecs(
                        alpha=0.9,
                        sigma=1.0,
                        gamma1=0.3,
                        gamma2=0.4,
                        N=100,  # <- total population size
                    )

            init_cond (np.array): vector with five numbers for the initial conditions
                The order should be exactly this: [S E I1 I2 R].
                example:
                    init_cond = [96, 0, 2, 2, 0]

            times (np.array): array with times to predict for
            betas (np.array): array with betas to predict for
            thetas (np.array): optional array with a term indicating size of SEIIR
                adjustment by day. If None, defaults to an adjustment of 0. If
                not None, must have the same dimensions as betas.
            dt (float): Optional, step of the solver. I left it sticking outside
                in case it works slow, so you can decrease it from the IHME pipeline.

        Returns:
            result (DataFrame):  a dataframe with columns ["S", "E", "I1", "I2", "R", "t", "beta"]
            where t and beta are times and beta which were provided, and others are solution
            of the ODE
        """
        forecaster = ODERunner(model_specs, init_cond, dt=dt)
        return forecaster.get_solution(times, beta=betas, theta=thetas)


# FIXME: The only "modeling" code shared between the stages.  Where to put it?
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
