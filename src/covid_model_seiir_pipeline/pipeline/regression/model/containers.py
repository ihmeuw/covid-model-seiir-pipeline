"""Containers for regression data."""
from dataclasses import dataclass
from typing import Dict, Iterator, Tuple, Union

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
    population: pd.Series

    alpha: pd.Series
    sigma: pd.Series
    gamma1: pd.Series
    gamma2: pd.Series

    vaccines_immune: pd.Series
    vaccines_other: pd.Series

    def to_dict(self) -> Dict[str, pd.Series]:
        return utilities.asdict(self)

    def to_df(self) -> pd.DataFrame:
        return pd.concat([v.rename(k) for k, v in self.to_dict().items()], axis=1)

    def __iter__(self) -> Iterator[Tuple[int, 'ODEParameters']]:
        location_ids = self.population.reset_index().location_id.unique()
        for location_id in location_ids:
            yield ODEParameters(**{
                key: value.loc[location_id] for key, value in self.to_dict().items()
            })



@dataclass
class HospitalCensusData:
    hospital_census: pd.Series
    icu_census: pd.Series
    ventilator_census: pd.Series

    def to_dict(self) -> Dict[str, pd.Series]:
        return utilities.asdict(self)


@dataclass
class HospitalMetrics:
    hospital_admissions: pd.Series
    hospital_census: pd.Series
    icu_admissions: pd.Series
    icu_census: pd.Series
    ventilator_census: pd.Series

    def to_dict(self) -> Dict[str, pd.Series]:
        return utilities.asdict(self)


@dataclass
class HospitalCorrectionFactors:
    hospital_census: pd.Series
    icu_census: pd.Series
    ventilator_census: pd.Series

    def to_dict(self) -> Dict[str, pd.Series]:
        return utilities.asdict(self)
