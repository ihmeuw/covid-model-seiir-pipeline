"""Containers for ode fit data."""
from dataclasses import dataclass
from typing import Dict, List, Iterator, Tuple

import pandas as pd

from covid_model_seiir_pipeline.lib import (
    utilities,
)


@dataclass
class ODEParameters:
    """Parameter container for the ODE fit of beta.

    This should be constructed such that each parameter series has a shared
    location-date index.
    """

    population: pd.Series
    new_e: pd.Series

    alpha: pd.Series
    sigma: pd.Series
    gamma1: pd.Series
    gamma2: pd.Series

    kappa: pd.Series
    rho: pd.Series
    phi: pd.Series
    pi: pd.Series
    rho_variant: pd.Series
    p_cross_immune: pd.Series

    vaccines_unprotected: pd.Series
    vaccines_protected_wild_type: pd.Series
    vaccines_protected_all_types: pd.Series
    vaccines_immune_wild_type: pd.Series
    vaccines_immune_all_types: pd.Series

    def to_dict(self) -> Dict[str, pd.Series]:
        return utilities.asdict(self)

    def to_df(self) -> pd.DataFrame:
        return pd.concat([v.rename(k) for k, v in self.to_dict().items() if isinstance(v, pd.Series)], axis=1)

    def get_vaccinations(self, vaccine_types: List[str]) -> pd.DataFrame:
        vaccine_type_map = {
            'u': self.vaccines_unprotected,
            'p': self.vaccines_protected_wild_type,
            'pa': self.vaccines_protected_all_types,
            'm': self.vaccines_immune_wild_type,
            'ma': self.vaccines_immune_all_types,
        }
        vaccinations = pd.DataFrame({
            k: vaccine_type_map[k].rename(k) for k in vaccine_types
        })
        return vaccinations

    def reindex(self, index: pd.Index) -> 'ODEParameters':
        return ODEParameters(
            **{key: value.reindex(index) for key, value in self.to_dict().items() if isinstance(value, pd.Series)},
        )

    def __iter__(self) -> Iterator[Tuple[int, 'ODEParameters']]:
        location_ids = self.population.reset_index().location_id.unique()
        for location_id in location_ids:
            loc_parameters = ODEParameters(
                **{key: value.loc[location_id]
                   for key, value in self.to_dict().items() if isinstance(value, pd.Series)},
            )
            yield location_id, loc_parameters
