from typing import Tuple

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    math,
    ode,
)
from covid_model_seiir_pipeline.pipeline.forecasting.model.containers import (
    Indices,
)


def build_model_parameters(indices: Indices,
                           ode_parameters: pd.DataFrame,
                           counterfactual_beta: pd.Series,
                           rhos: pd.DataFrame,
                           vaccine_data: pd.DataFrame) -> ode.ForecastParameters:
    # These are all the same by draw.  Just broadcasting them over a new index.
    ode_params = {
        param: pd.Series(ode_parameters[param].mean(), index=indices.full, name=param)
        for param in ['alpha', 'sigma', 'gamma1', 'gamma2', 'pi', 'chi']
    }

    beta, beta_wild, beta_variant, beta_hat, rho, rho_variant, rho_b1617, rho_total = get_betas_and_prevalences(
        indices,
        counterfactual_beta,
        rhos,
        ode_parameters['kappa'].mean(),
        ode_parameters['phi'].mean(),
        ode_parameters['psi'].mean(),
    )

    vaccine_data = vaccine_data.reindex(indices.full, fill_value=0)
    adjusted_vaccinations = math.adjust_vaccinations(vaccine_data)

    return ode.ForecastParameters(
        **ode_params,
        beta=beta,
        beta_wild=beta_wild,
        beta_variant=beta_variant,
        beta_hat=beta_hat,
        rho=rho,
        rho_variant=rho_variant,
        rho_b1617=rho_b1617,
        rho_total=rho_total,
        **adjusted_vaccinations,
    )


def get_betas_and_prevalences(indices: Indices,
                              counterfactual_beta: pd.Series,
                              rhos: pd.DataFrame,
                              kappa: float,
                              phi: float,
                              psi: float) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series,
                                                   pd.Series, pd.Series, pd.Series, pd.Series]:
    rhos = rhos.reindex(indices.full).fillna(method='ffill')

    beta_hat = counterfactual_beta.copy()
    beta_hat.loc[:] = np.nan
    beta = counterfactual_beta.copy()
    beta_wild = beta * (1 + kappa * rhos.rho)
    beta_variant = beta * (1 + kappa * (phi * (1 - rhos.rho_b1617) + rhos.rho_b1617 * psi))

    return (beta, beta_wild, beta_variant, beta_hat,
            rhos.rho, rhos.rho_variant, rhos.rho_b1617, rhos.rho_total)
