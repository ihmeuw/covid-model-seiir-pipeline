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
    AGE_SCALARS_NAMES,
    VACCINE_STATUS_NAMES,
    ETA_NAMES,
    CHI_NAMES,
)


class Parameters:

    def __init__(self,
                 base_parameters: pd.DataFrame,
                 vaccinations: pd.DataFrame,
                 age_scalars: pd.DataFrame,
                 etas: pd.DataFrame,
                 phis: pd.DataFrame):
        self.base_parameters = base_parameters.loc[:, PARAMETERS_NAMES]

        field_names = self._make_risk_group_fields(VACCINE_STATUS_NAMES[1:])
        self.vaccinations = vaccinations.loc[:, field_names]

        field_names = self._make_risk_group_fields(AGE_SCALARS_NAMES)
        self.age_scalars = age_scalars.loc[:, field_names]

        field_names = self._make_risk_group_fields(ETA_NAMES)
        self.etas = etas.loc[:, field_names]

        self.phis = phis.loc[:, CHI_NAMES]

    @staticmethod
    def _make_risk_group_fields(measures) -> List[str]:
        return [f'{measure}_{risk_group}' for risk_group, measure
                in itertools.product(RISK_GROUP_NAMES, measures)]

    def to_dict(self) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        return {k: v for k, v in self.__dict__.items() if isinstance(v, (pd.DataFrame, pd.Series))}
