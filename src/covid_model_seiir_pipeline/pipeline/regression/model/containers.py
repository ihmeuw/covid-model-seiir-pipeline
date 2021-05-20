"""Containers for regression data."""
from dataclasses import dataclass
from typing import Dict, List, Iterator, Tuple, Union

import pandas as pd

from covid_model_seiir_pipeline.lib import (
    utilities,
)


@dataclass
class RatioData:
    infection_to_death: int
    infection_to_admission: int
    infection_to_case: int

    ifr: pd.Series
    ifr_hr: pd.Series
    ifr_lr: pd.Series
    ihr: pd.Series
    idr: pd.Series

    def to_dict(self) -> Dict[str, Union[int, pd.Series]]:
        return utilities.asdict(self)


@dataclass
class ODEParameters:
    """Parameter container for the ODE fit of beta.

    This should be constructed such that each parameter series has a shared
    location-date index.
    """
    # Sub-populations
    population_low_risk: pd.Series
    population_high_risk: pd.Series

    # Core parameters
    alpha: pd.Series
    sigma: pd.Series
    gamma1: pd.Series
    gamma2: pd.Series

    # Transmission intensity
    new_e: pd.Series
    kappa: pd.Series
    phi: pd.Series
    psi: pd.Series

    # Variant prevalences
    rho: pd.Series
    rho_variant: pd.Series
    rho_b1617: pd.Series

    # Escape variant initialization
    pi: pd.Series

    # Cross-variant immunity
    chi: pd.Series

    # Vaccine parameters
    unprotected_lr: pd.Series
    protected_wild_type_lr: pd.Series
    protected_all_types_lr: pd.Series
    immune_wild_type_lr: pd.Series
    immune_all_types_lr: pd.Series

    unprotected_hr: pd.Series
    protected_wild_type_hr: pd.Series
    protected_all_types_hr: pd.Series
    immune_wild_type_hr: pd.Series
    immune_all_types_hr: pd.Series

    def reindex(self, index: pd.Index) -> 'ODEParameters':
        return ODEParameters(
            **{key: value.reindex(index) for key, value in self.to_dict().items() if isinstance(value, pd.Series)},
        )

    def to_dict(self) -> Dict[str, pd.Series]:
        return {k: v.rename(k) for k, v in utilities.asdict(self).items()}

    def to_df(self) -> pd.DataFrame:
        return pd.concat(self.to_dict().values(), axis=1)

    def get_vaccinations(self, vaccine_types: List[str], risk_group: str) -> pd.DataFrame:
        vaccine_type_map = {
            'u': 'unprotected',
            'p': 'protected_wild_type',
            'pa': 'protected_all_types',
            'm': 'immune_wild_type',
            'ma': 'immune_all_types',
        }
        vaccinations = []
        for vaccine_type in vaccine_types:
            attr = f'{vaccine_type_map[vaccine_type]}_{risk_group}'
            vaccinations.append(getattr(self, attr).rename(attr))

        return pd.concat(vaccinations, axis=1)

    def __iter__(self) -> Iterator[Tuple[int, 'ODEParameters']]:
        location_ids = self.population_low_risk.reset_index().location_id.unique()
        for location_id in location_ids:
            loc_parameters = ODEParameters(
                **{key: value.loc[location_id]
                   for key, value in self.to_dict().items() if isinstance(value, pd.Series)},
            )
            yield location_id, loc_parameters


@dataclass
class HospitalCensusData:
    hospital_census: pd.Series
    icu_census: pd.Series

    def to_dict(self) -> Dict[str, pd.Series]:
        return utilities.asdict(self)

    def to_df(self):
        return pd.concat([v.rename(k) for k, v in self.to_dict().items()], axis=1)


@dataclass
class HospitalMetrics:
    hospital_admissions: pd.Series
    hospital_census: pd.Series
    icu_admissions: pd.Series
    icu_census: pd.Series

    def to_dict(self) -> Dict[str, pd.Series]:
        return utilities.asdict(self)

    def to_df(self):
        return pd.concat([v.rename(k) for k, v in self.to_dict().items()], axis=1)


@dataclass
class HospitalCorrectionFactors:
    hospital_census: pd.Series
    icu_census: pd.Series

    def to_dict(self) -> Dict[str, pd.Series]:
        return utilities.asdict(self)

    def to_df(self):
        return pd.concat([v.rename(k) for k, v in self.to_dict().items()], axis=1)
