"""Static definitions for data containers.

This code is automatically generated by generator/make_containers.py

Any manual changes will be lost.
"""
from dataclasses import (
    dataclass,
)
from typing import (
    Dict,
    List,
    Iterator,
    Tuple,
)

import pandas as pd

from covid_model_seiir_pipeline.lib import (
    utilities,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    PARAMETERS_NAMES,
    ETA_NAMES,
)


@dataclass(repr=False, eq=False)
class Parameters:
    alpha_all: pd.Series
    sigma_all: pd.Series
    gamma_all: pd.Series
    pi_all: pd.Series
    new_e_all: pd.Series
    beta_all: pd.Series
    kappa_none: pd.Series
    kappa_ancestral: pd.Series
    kappa_alpha: pd.Series
    kappa_beta: pd.Series
    kappa_gamma: pd.Series
    kappa_delta: pd.Series
    kappa_other: pd.Series
    kappa_omega: pd.Series
    rho_none: pd.Series
    rho_ancestral: pd.Series
    rho_alpha: pd.Series
    rho_beta: pd.Series
    rho_gamma: pd.Series
    rho_delta: pd.Series
    rho_other: pd.Series
    rho_omega: pd.Series

    vaccinations_lr: pd.Series
    boosters_lr: pd.Series
    vaccinations_hr: pd.Series
    boosters_hr: pd.Series

    eta_unvaccinated_none_lr: pd.Series
    eta_unvaccinated_ancestral_lr: pd.Series
    eta_unvaccinated_alpha_lr: pd.Series
    eta_unvaccinated_beta_lr: pd.Series
    eta_unvaccinated_gamma_lr: pd.Series
    eta_unvaccinated_delta_lr: pd.Series
    eta_unvaccinated_other_lr: pd.Series
    eta_unvaccinated_omega_lr: pd.Series
    eta_vaccinated_none_lr: pd.Series
    eta_vaccinated_ancestral_lr: pd.Series
    eta_vaccinated_alpha_lr: pd.Series
    eta_vaccinated_beta_lr: pd.Series
    eta_vaccinated_gamma_lr: pd.Series
    eta_vaccinated_delta_lr: pd.Series
    eta_vaccinated_other_lr: pd.Series
    eta_vaccinated_omega_lr: pd.Series
    eta_booster_none_lr: pd.Series
    eta_booster_ancestral_lr: pd.Series
    eta_booster_alpha_lr: pd.Series
    eta_booster_beta_lr: pd.Series
    eta_booster_gamma_lr: pd.Series
    eta_booster_delta_lr: pd.Series
    eta_booster_other_lr: pd.Series
    eta_booster_omega_lr: pd.Series
    eta_unvaccinated_none_hr: pd.Series
    eta_unvaccinated_ancestral_hr: pd.Series
    eta_unvaccinated_alpha_hr: pd.Series
    eta_unvaccinated_beta_hr: pd.Series
    eta_unvaccinated_gamma_hr: pd.Series
    eta_unvaccinated_delta_hr: pd.Series
    eta_unvaccinated_other_hr: pd.Series
    eta_unvaccinated_omega_hr: pd.Series
    eta_vaccinated_none_hr: pd.Series
    eta_vaccinated_ancestral_hr: pd.Series
    eta_vaccinated_alpha_hr: pd.Series
    eta_vaccinated_beta_hr: pd.Series
    eta_vaccinated_gamma_hr: pd.Series
    eta_vaccinated_delta_hr: pd.Series
    eta_vaccinated_other_hr: pd.Series
    eta_vaccinated_omega_hr: pd.Series
    eta_booster_none_hr: pd.Series
    eta_booster_ancestral_hr: pd.Series
    eta_booster_alpha_hr: pd.Series
    eta_booster_beta_hr: pd.Series
    eta_booster_gamma_hr: pd.Series
    eta_booster_delta_hr: pd.Series
    eta_booster_other_hr: pd.Series
    eta_booster_omega_hr: pd.Series

    natural_waning_distribution: pd.Series
    phi: pd.DataFrame

    def get_params(self) -> pd.DataFrame:
        return pd.concat([v.rename(k) for k, v in utilities.asdict(self).items() if k in PARAMETERS_NAMES], axis=1)
        
    def get_vaccinations(self) -> pd.DataFrame:
        return pd.concat([v.rename(k) for k, v in utilities.asdict(self).items() 
                          if 'vaccinations' in k or 'boosters' in k], axis=1)    

    def get_etas(self) -> pd.DataFrame:
        return pd.concat([v.rename(k) for k, v in utilities.asdict(self).items() 
                          if '_'.join(k.split('_')[1:-1]) in ETA_NAMES], axis=1)
        
        
    def to_dfs(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
        return self.get_params(), self.get_vaccinations(), self.get_etas(), self.natural_waning_distribution, self.phi
        
    