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
    eta_none: pd.Series
    eta_ancestral: pd.Series
    eta_alpha: pd.Series
    eta_beta: pd.Series
    eta_gamma: pd.Series
    eta_delta: pd.Series
    eta_other: pd.Series
    eta_omega: pd.Series

    vaccinations_lr: pd.Series
    boosters_lr: pd.Series
    vaccinations_hr: pd.Series
    boosters_hr: pd.Series

    iota: pd.DataFrame

    def to_dict(self) -> Dict[str, pd.Series]:
        return {k: v.rename(k) for k, v in utilities.asdict(self).items() if k != 'iota'}

    def to_df(self) -> pd.DataFrame:
        return pd.concat(self.to_dict().values(), axis=1)
    