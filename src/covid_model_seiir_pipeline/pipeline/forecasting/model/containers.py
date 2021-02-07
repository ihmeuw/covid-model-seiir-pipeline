from dataclasses import dataclass
from typing import Dict, List, Union

import pandas as pd

from covid_model_seiir_pipeline.lib import (
    utilities,
)

# This is just exposing these containers from this namespace so we're not
# importing from the regression stage everywhere.
from covid_model_seiir_pipeline.pipeline.regression.model.containers import (
    HospitalFatalityRatioData,
    HospitalCensusData,
    HospitalMetrics,
    HospitalCorrectionFactors,
)


@dataclass
class OutputMetrics:
    components: pd.DataFrame
    infections: pd.Series
    deaths: pd.DataFrame
    r_controlled: pd.Series
    r_effective: pd.Series
    herd_immunity: pd.Series
    total_susceptible: pd.Series
    total_immune: pd.Series

    def to_dict(self) -> Dict[str, pd.DataFrame]:
        return utilities.asdict(self)


@dataclass
class CompartmentInfo:
    compartments: List[str]
    group_suffixes: List[str]

    def to_dict(self) -> Dict[str, List[str]]:
        return utilities.asdict(self)


@dataclass
class ScenarioData:
    vaccinations: Union[pd.DataFrame, None]
    percent_mandates: Union[pd.DataFrame, None]
    mandate_effects: Union[pd.DataFrame, None]

    def to_dict(self) -> Dict[str, Union[pd.DataFrame, None]]:
        return utilities.asdict(self)


@dataclass
class VariantScalars:
    beta: pd.Series
    ifr: pd.Series

    def to_dict(self) -> Dict[str, pd.Series]:
        return utilities.asdict(self)
