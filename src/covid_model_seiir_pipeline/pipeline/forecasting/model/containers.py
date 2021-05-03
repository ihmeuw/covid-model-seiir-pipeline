from dataclasses import dataclass
from typing import Dict, List, Union

import pandas as pd

from covid_model_seiir_pipeline.lib import (
    utilities,
)

# This is just exposing these containers from this namespace so we're not
# importing from the regression stage everywhere.
from covid_model_seiir_pipeline.pipeline.regression.model.containers import (
    RatioData,
    HospitalCensusData,
    HospitalMetrics,
    HospitalCorrectionFactors,
)


class Indices:
    """Abstraction for building square datasets."""

    def __init__(self,
                 past_start_dates: pd.Series,
                 forecast_start_dates: pd.Series,
                 forecast_end_dates: pd.Series):
        self._past_index = self._build_index(past_start_dates, forecast_start_dates, pd.Timedelta(days=1))
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
class ModelParameters:
    # Core parameters
    alpha: pd.Series
    sigma: pd.Series
    gamma1: pd.Series
    gamma2: pd.Series

    # Transmission intensity
    beta: pd.Series
    beta_wild: pd.Series
    beta_variant: pd.Series
    beta_hat: pd.Series

    # Variant prevalences
    rho: pd.Series
    rho_variant: pd.Series
    rho_total: pd.Series

    # Escape variant initialization
    pi: pd.Series

    # Cross-variant immunity
    chi: pd.Series

    # Theta parameters
    theta_plus: pd.Series
    theta_minus: pd.Series

    # Vaccine parameters
    unprotected_lr: pd.Series
    protected_wild_type_lr: pd.Series
    protected_all_types_lr: pd.Series
    immune_wild_type_lr: pd.Series
    immune_all_types_lr: pd.Series

    unprotected_hr: pd.Series
    protected_wild_type_hr: pd.Series
    protected_all_types_hr: pd.Series
    immune_wild_type_hr: pd.Series
    immune_all_types_hr: pd.Series

    def with_index(self, index: pd.MultiIndex):
        return ModelParameters(**{
            parameter_name: parameter.loc[index] for parameter_name, parameter in self.to_dict().items()
        })

    def to_dict(self) -> Dict[str, pd.Series]:
        return {k: v.rename(k) for k, v in utilities.asdict(self).items()}

    def to_df(self) -> pd.DataFrame:
        return pd.concat(self.to_dict().values(), axis=1)


@dataclass
class PostprocessingParameters:
    past_compartments: pd.DataFrame

    past_infections: pd.Series
    past_deaths: pd.Series

    infection_to_death: int
    infection_to_admission: int
    infection_to_case: int

    ifr: pd.Series
    ifr_hr: pd.Series
    ifr_lr: pd.Series
    ihr: pd.Series
    idr: pd.Series

    hospital_census: pd.Series
    icu_census: pd.Series
    ventilator_census: pd.Series

    def to_dict(self) -> Dict[str, Union[int, pd.Series, pd.DataFrame]]:
        return utilities.asdict(self)


@dataclass
class SystemMetrics:
    modeled_infections_wild: pd.Series
    modeled_infections_variant: pd.Series
    modeled_infections_total: pd.Series
    modeled_infected_total = pd.Series

    variant_prevalence: pd.Series
    natural_immunity_breakthrough: pd.Series
    vaccine_breakthrough: pd.Series
    proportion_cross_immune: pd.Series

    modeled_deaths_wild: pd.Series
    modeled_deaths_variant: pd.Series
    modeled_deaths_total: pd.Series

    vaccinations_protected_wild: pd.Series
    vaccinations_protected_all: pd.Series
    vaccinations_immune_wild: pd.Series
    vaccinations_immune_all: pd.Series
    vaccinations_effective: pd.Series
    vaccinations_ineffective: pd.Series

    total_susceptible_wild: pd.Series
    total_susceptible_variant: pd.Series
    total_susceptible_variant_only: pd.Series
    total_infectious_wild: pd.Series
    total_infectious_variant: pd.Series
    total_immune_wild: pd.Series
    total_immune_variant: pd.Series

    total_population: pd.Series

    beta: pd.Series
    beta_wild: pd.Series
    beta_variant: pd.Series

    def to_dict(self) -> Dict[str, pd.Series]:
        return utilities.asdict(self)

    def to_df(self) -> pd.DataFrame:
        return pd.concat([v.rename(k) for k, v in self.to_dict().items()], axis=1)


@dataclass
class OutputMetrics:
    # observed + modeled
    infections: pd.Series
    cases: pd.Series
    hospital_admissions: pd.Series
    hospital_census: pd.Series
    icu_admissions: pd.Series
    icu_census: pd.Series
    ventilator_census: pd.Series
    deaths: pd.Series

    # Other stuff
    r_controlled_wild: pd.Series
    r_effective_wild: pd.Series
    r_controlled_variant: pd.Series
    r_effective_variant: pd.Series
    r_effective: pd.Series

    def to_dict(self) -> Dict[str, pd.Series]:
        return utilities.asdict(self)

    def to_df(self) -> pd.DataFrame:
        out_list = []
        for k, v in self.to_dict().items():
            v = v.rename(k)
            if k == 'deaths':
                v = v.reset_index(level='observed')
            out_list.append(v)
        return pd.concat(out_list, axis=1)
