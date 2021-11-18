"""Static definitions for data containers."""
import itertools
from typing import (
    Dict,
    List,
    Union,
)

import pandas as pd

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    RISK_GROUP_NAMES,
    PARAMETERS_NAMES,
    RATES_NAMES,
    ETA_NAMES,
    CHI_NAMES,
)


class Parameters:

    def __init__(self,
                 base_parameters: pd.DataFrame,
                 vaccinations: pd.DataFrame,
                 rates: pd.DataFrame,
                 eta_infections: pd.DataFrame,
                 eta_deaths: pd.DataFrame,
                 eta_admissions: pd.DataFrame,
                 eta_cases: pd.DataFrame,
                 natural_waning_distribution: pd.DataFrame,
                 phi_infections: pd.DataFrame,
                 phi_deaths: pd.DataFrame,
                 phi_admissions: pd.DataFrame,
                 phi_cases: pd.DataFrame):
        self.base_parameters = base_parameters.loc[:, PARAMETERS_NAMES]

        field_names = self._make_risk_group_fields(['vaccinations', 'boosters'])
        self.vaccinations = vaccinations.loc[:, field_names]

        field_names = self._make_risk_group_fields(RATES_NAMES)
        self.rates = rates.loc[:, field_names]

        field_names = self._make_risk_group_fields(ETA_NAMES)
        self.eta_infections = eta_infections.loc[:, field_names]
        self.eta_deaths = eta_deaths.loc[:, field_names]
        self.eta_admissions = eta_admissions.loc[:, field_names]
        self.eta_cases = eta_cases.loc[:, field_names]

        self.natural_waning_distribution = natural_waning_distribution

        self.phi = phi.loc[:, CHI_NAMES]


    @staticmethod
    def _make_risk_group_fields(measures) -> List[str]:
        return [f'{measure}_{risk_group}' for risk_group, measure
                in itertools.product(RISK_GROUP_NAMES, measures)]

    def to_dict(self) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        return {k: v for k, v in self.__dict__ if isinstance(v, (pd.DataFrame, pd.Series))}
