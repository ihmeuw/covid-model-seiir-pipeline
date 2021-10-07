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
    PHI_NAMES,
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
    vaccinations_hr: pd.Series
    boosters_lr: pd.Series
    boosters_hr: pd.Series

    eta_unvaccinated_none_lr: pd.Series
    eta_unvaccinated_none_hr: pd.Series
    eta_unvaccinated_ancestral_lr: pd.Series
    eta_unvaccinated_ancestral_hr: pd.Series
    eta_unvaccinated_alpha_lr: pd.Series
    eta_unvaccinated_alpha_hr: pd.Series
    eta_unvaccinated_beta_lr: pd.Series
    eta_unvaccinated_beta_hr: pd.Series
    eta_unvaccinated_gamma_lr: pd.Series
    eta_unvaccinated_gamma_hr: pd.Series
    eta_unvaccinated_delta_lr: pd.Series
    eta_unvaccinated_delta_hr: pd.Series
    eta_unvaccinated_other_lr: pd.Series
    eta_unvaccinated_other_hr: pd.Series
    eta_unvaccinated_omega_lr: pd.Series
    eta_unvaccinated_omega_hr: pd.Series
    eta_vaccinated_none_lr: pd.Series
    eta_vaccinated_none_hr: pd.Series
    eta_vaccinated_ancestral_lr: pd.Series
    eta_vaccinated_ancestral_hr: pd.Series
    eta_vaccinated_alpha_lr: pd.Series
    eta_vaccinated_alpha_hr: pd.Series
    eta_vaccinated_beta_lr: pd.Series
    eta_vaccinated_beta_hr: pd.Series
    eta_vaccinated_gamma_lr: pd.Series
    eta_vaccinated_gamma_hr: pd.Series
    eta_vaccinated_delta_lr: pd.Series
    eta_vaccinated_delta_hr: pd.Series
    eta_vaccinated_other_lr: pd.Series
    eta_vaccinated_other_hr: pd.Series
    eta_vaccinated_omega_lr: pd.Series
    eta_vaccinated_omega_hr: pd.Series
    eta_booster_none_lr: pd.Series
    eta_booster_none_hr: pd.Series
    eta_booster_ancestral_lr: pd.Series
    eta_booster_ancestral_hr: pd.Series
    eta_booster_alpha_lr: pd.Series
    eta_booster_alpha_hr: pd.Series
    eta_booster_beta_lr: pd.Series
    eta_booster_beta_hr: pd.Series
    eta_booster_gamma_lr: pd.Series
    eta_booster_gamma_hr: pd.Series
    eta_booster_delta_lr: pd.Series
    eta_booster_delta_hr: pd.Series
    eta_booster_other_lr: pd.Series
    eta_booster_other_hr: pd.Series
    eta_booster_omega_lr: pd.Series
    eta_booster_omega_hr: pd.Series

    phi_none_none: pd.Series
    phi_none_ancestral: pd.Series
    phi_none_alpha: pd.Series
    phi_none_beta: pd.Series
    phi_none_gamma: pd.Series
    phi_none_delta: pd.Series
    phi_none_other: pd.Series
    phi_none_omega: pd.Series
    phi_ancestral_none: pd.Series
    phi_ancestral_ancestral: pd.Series
    phi_ancestral_alpha: pd.Series
    phi_ancestral_beta: pd.Series
    phi_ancestral_gamma: pd.Series
    phi_ancestral_delta: pd.Series
    phi_ancestral_other: pd.Series
    phi_ancestral_omega: pd.Series
    phi_alpha_none: pd.Series
    phi_alpha_ancestral: pd.Series
    phi_alpha_alpha: pd.Series
    phi_alpha_beta: pd.Series
    phi_alpha_gamma: pd.Series
    phi_alpha_delta: pd.Series
    phi_alpha_other: pd.Series
    phi_alpha_omega: pd.Series
    phi_beta_none: pd.Series
    phi_beta_ancestral: pd.Series
    phi_beta_alpha: pd.Series
    phi_beta_beta: pd.Series
    phi_beta_gamma: pd.Series
    phi_beta_delta: pd.Series
    phi_beta_other: pd.Series
    phi_beta_omega: pd.Series
    phi_gamma_none: pd.Series
    phi_gamma_ancestral: pd.Series
    phi_gamma_alpha: pd.Series
    phi_gamma_beta: pd.Series
    phi_gamma_gamma: pd.Series
    phi_gamma_delta: pd.Series
    phi_gamma_other: pd.Series
    phi_gamma_omega: pd.Series
    phi_delta_none: pd.Series
    phi_delta_ancestral: pd.Series
    phi_delta_alpha: pd.Series
    phi_delta_beta: pd.Series
    phi_delta_gamma: pd.Series
    phi_delta_delta: pd.Series
    phi_delta_other: pd.Series
    phi_delta_omega: pd.Series
    phi_other_none: pd.Series
    phi_other_ancestral: pd.Series
    phi_other_alpha: pd.Series
    phi_other_beta: pd.Series
    phi_other_gamma: pd.Series
    phi_other_delta: pd.Series
    phi_other_other: pd.Series
    phi_other_omega: pd.Series
    phi_omega_none: pd.Series
    phi_omega_ancestral: pd.Series
    phi_omega_alpha: pd.Series
    phi_omega_beta: pd.Series
    phi_omega_gamma: pd.Series
    phi_omega_delta: pd.Series
    phi_omega_other: pd.Series
    phi_omega_omega: pd.Series

    def get_params(self) -> pd.DataFrame:
        return pd.concat([v.rename(k) for k, v in utilities.asdict(self).items() if k in PARAMETERS_NAMES], axis=1)
        
    def get_vaccinations(self) -> pd.DataFrame:
        return pd.concat([v.rename(k) for k, v in utilities.asdict(self).items() 
                          if 'vaccinations' in k or 'boosters' in k], axis=1)    

    def get_etas(self) -> pd.DataFrame:
        return pd.concat([v.rename(k) for k, v in utilities.asdict(self).items() 
                          if '_'.join(k.split('_')[1:-1]) in ETA_NAMES], axis=1)
        
    def get_phis(self) -> pd.DataFrame:
        return pd.concat([v.rename(k) for k, v in utilities.asdict(self).items() if k in PHI_NAMES], axis=1)
        
    def to_dfs(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.get_params(), self.get_vaccinations(), self.get_etas(), self.get_phis()
        
    