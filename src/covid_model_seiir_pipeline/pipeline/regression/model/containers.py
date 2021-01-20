"""Containers for regression data."""
from dataclasses import dataclass
from typing import Dict

import pandas as pd

from covid_model_seiir_pipeline.lib import (
    utilities,
)


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
