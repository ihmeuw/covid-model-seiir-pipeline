"""Containers for regression data."""
from dataclasses import dataclass
from typing import Dict, Union

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
    alpha: float
    sigma: float
    gamma1: float
    gamma2: float

    def to_dict(self) -> Dict[str, Union[int, float]]:
        return utilities.asdict(self)


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
