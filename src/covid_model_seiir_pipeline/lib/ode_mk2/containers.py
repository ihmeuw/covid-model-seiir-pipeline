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
    VARIANT_NAMES,
)


class Parameters:

    def __init__(self,
                 base_parameters: pd.DataFrame,
                 vaccinations: pd.DataFrame,
                 rates: pd.DataFrame,
                 eta_infection: pd.DataFrame,
                 eta_death: pd.DataFrame,
                 eta_admission: pd.DataFrame,
                 eta_case: pd.DataFrame,
                 natural_waning_infection: pd.DataFrame,
                 natural_waning_death: pd.DataFrame,
                 natural_waning_admission: pd.DataFrame,
                 natural_waning_case: pd.DataFrame,
                 phi: pd.DataFrame,
                 ):
        self.base_parameters = base_parameters.loc[:, PARAMETERS_NAMES]

        field_names = self._make_risk_group_fields(['vaccinations', 'boosters'])
        self.vaccinations = vaccinations.loc[:, field_names]

        field_names = self._make_risk_group_fields(RATES_NAMES)
        self.rates = rates.loc[:, field_names]

        field_names = self._make_risk_group_fields(ETA_NAMES)
        self.eta_infection = eta_infection.loc[:, field_names]
        self.eta_death = eta_death.loc[:, field_names]
        self.eta_admission = eta_admission.loc[:, field_names]
        self.eta_case = eta_case.loc[:, field_names]

        self.natural_waning_infection = natural_waning_infection
        self.natural_waning_death = natural_waning_death
        self.natural_waning_admission = natural_waning_admission
        self.natural_waning_case = natural_waning_case

        self.phi = phi.loc[VARIANT_NAMES, VARIANT_NAMES]

    @staticmethod
    def _make_risk_group_fields(measures) -> List[str]:
        return [f'{measure}_{risk_group}' for risk_group, measure
                in itertools.product(RISK_GROUP_NAMES, measures)]

    def to_dict(self) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        return {k: v for k, v in self.__dict__ if isinstance(v, (pd.DataFrame, pd.Series))}
