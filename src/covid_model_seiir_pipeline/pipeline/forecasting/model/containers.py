from dataclasses import dataclass
from typing import Dict, List, Union

import pandas as pd

from covid_model_seiir_pipeline.lib import (
    utilities,
)


class Indices:
    """Abstraction for building square datasets."""

    def __init__(self,
                 past_start_dates: pd.Series,
                 beta_fit_end_dates: pd.Series,
                 forecast_start_dates: pd.Series,
                 forecast_end_dates: pd.Series):
        self._past_index = self._build_index(past_start_dates, forecast_start_dates, pd.Timedelta(days=1))
        self._beta_fit_index = self._build_index(past_start_dates, beta_fit_end_dates)
        self._future_index = self._build_index(forecast_start_dates, forecast_end_dates)
        self._initial_condition_index = (
            forecast_start_dates
            .reset_index()
            .set_index(['location_id', 'date'])
            .sort_index()
            .index
        )
        self._full_index = self._build_index(past_start_dates, forecast_end_dates)

    @property
    def past(self) -> pd.MultiIndex:
        """Location-date index for the past."""
        return self._past_index.copy()

    @property
    def beta_fit(self) -> pd.MultiIndex:
        """Location-date index for the past."""
        return self._beta_fit_index.copy()

    @property
    def future(self) -> pd.MultiIndex:
        """Location-date index for the future."""
        return self._future_index.copy()

    @property
    def initial_condition(self) -> pd.MultiIndex:
        """Location-date index for the initial condition.

        This index has one date per location.
        """
        return self._initial_condition_index.copy()

    @property
    def full(self) -> pd.MultiIndex:
        """Location-date index for the full time series, past and future."""
        return self._full_index.copy()

    @staticmethod
    def _build_index(start: pd.Series,
                     end: pd.Series,
                     end_offset: pd.Timedelta = pd.Timedelta(days=0)) -> pd.MultiIndex:
        index = (pd.concat([start.rename('start'), end.rename('end')], axis=1)
                 .groupby('location_id')
                 .apply(lambda x: pd.date_range(x.iloc[0, 0], x.iloc[0, 1] - end_offset))
                 .explode()
                 .rename('date')
                 .reset_index()
                 .set_index(['location_id', 'date'])
                 .sort_index()
                 .index)
        return index


@dataclass
class PostprocessingParameters:
    past_infections: pd.Series
    past_deaths: pd.Series

    infection_to_death: int
    infection_to_admission: int
    infection_to_case: int

    ifr_scalar: float
    ihr_scalar: float

    ifr: pd.Series
    ifr_hr: pd.Series
    ifr_lr: pd.Series
    ihr: pd.Series
    idr: pd.Series

    hospital_census: pd.Series
    icu_census: pd.Series

    def to_dict(self) -> Dict[str, Union[int, pd.Series, pd.DataFrame]]:
        return utilities.asdict(self)

    @property
    def correction_factors_df(self) -> pd.DataFrame:
        return pd.concat([
            self.hospital_census.rename('hospital_census_correction_factor'),
            self.icu_census.rename('icu_census_correction_factor'),
        ], axis=1)
