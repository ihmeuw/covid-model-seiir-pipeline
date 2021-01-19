"""Containers for regression data."""
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import pandas as pd

from covid_model_seiir_pipeline.lib import (
    utilities,
)


@dataclass
class ODEProcessInput:
    df_dict: Dict[int, pd.DataFrame]
    col_date: str
    col_infections: str
    col_pop: str
    col_loc_id: str
    col_lag_days: str
    col_observed: str

    alpha: Tuple[float, float]
    sigma: Tuple[float, float]
    gamma1: Tuple[float, float]
    gamma2: Tuple[float, float]
    solver_dt: float
    day_shift: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        return utilities.asdict(self)


@dataclass
class HospitalFatalityRatioData:
    age_specific: pd.Series
    all_age: pd.Series

    def to_dict(self) -> Dict[str, pd.Series]:
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
