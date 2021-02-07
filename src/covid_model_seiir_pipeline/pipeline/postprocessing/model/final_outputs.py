from typing import Any, Callable, Dict, List, TYPE_CHECKING

import pandas as pd

from covid_model_seiir_pipeline.pipeline.postprocessing.model import aggregators, loaders, combiners

if TYPE_CHECKING:
    # The model subpackage is a library for the pipeline stage and shouldn't
    # explicitly depend on things outside the subpackage.
    from covid_model_seiir_pipeline.pipeline.postprocessing.data import PostprocessingDataInterface


class MeasureConfig:
    def __init__(self,
                 loader: Callable[[str, 'PostprocessingDataInterface', int], Any],
                 label: str,
                 splice: bool = True,
                 calculate_cumulative: bool = False,
                 cumulative_label: str = None,
                 aggregator: Callable = None):
        self.loader = loader
        self.label = label
        self.splice = splice
        self.calculate_cumulative = calculate_cumulative
        self.cumulative_label = cumulative_label
        self.aggregator = aggregator


class CompositeMeasureConfig:
    def __init__(self,
                 base_measures: Dict[str, MeasureConfig],
                 label: str,
                 combiner: Callable = None):
        self.base_measures = base_measures
        self.label = label
        self.combiner = combiner


class CovariateConfig:
    def __init__(self,
                 loader: Callable[[str, bool, str, 'PostprocessingDataInterface', int], List[pd.Series]],
                 label: str,
                 splice: bool = False,
                 time_varying: bool = False,
                 draw_level: bool = False,
                 aggregator: Callable = None):
        self.loader = loader
        self.label = label
        self.splice = splice
        self.time_varying = time_varying
        self.draw_level = draw_level
        self.aggregator = aggregator


class MiscellaneousConfig:
    def __init__(self,
                 loader: Callable[['PostprocessingDataInterface'], Any],
                 label: str,
                 is_table: bool = True,
                 is_cumulative: bool = False,
                 aggregator: Callable = None):
        self.loader = loader
        self.label = label
        self.is_table = is_table
        self.is_cumulative = is_cumulative
        self.aggregator = aggregator


MEASURES = {
    'deaths': MeasureConfig(
        loaders.load_deaths,
        'daily_deaths',
        calculate_cumulative=True,
        cumulative_label='cumulative_deaths',
        aggregator=aggregators.sum_aggregator,
    ),
    'infections': MeasureConfig(
        loaders.load_infections,
        'daily_infections',
        calculate_cumulative=True,
        cumulative_label='cumulative_infections',
        aggregator=aggregators.sum_aggregator,
    ),
    'r_controlled': MeasureConfig(
        loaders.load_r_controlled,
        'r_controlled',
    ),
    'r_effective': MeasureConfig(
        loaders.load_r_effective,
        'r_effective',
    ),
    'herd_immunity': MeasureConfig(
        loaders.load_herd_immunity,
        'herd_immunity',
    ),
    'total_susceptible': MeasureConfig(
        loaders.load_total_susceptible,
        'total_susceptible',
        aggregator=aggregators.sum_aggregator,
    ),
    'total_immune': MeasureConfig(
        loaders.load_total_immune,
        'total_immune',
        aggregator=aggregators.sum_aggregator,
    ),
    'hospital_admissions': MeasureConfig(
        loaders.load_hospital_admissions,
        'hospital_admissions',
        aggregator=aggregators.sum_aggregator,
    ),
    'icu_admissions': MeasureConfig(
        loaders.load_icu_admissions,
        'icu_admissions',
        aggregator=aggregators.sum_aggregator,
    ),
    'hospital_census': MeasureConfig(
        loaders.load_hospital_census,
        'hospital_census',
        aggregator=aggregators.sum_aggregator,
    ),
    'icu_census': MeasureConfig(
        loaders.load_icu_census,
        'icu_census',
        aggregator=aggregators.sum_aggregator,
    ),
    'ventilator_census': MeasureConfig(
        loaders.load_ventilator_census,
        'ventilator_census',
        aggregator=aggregators.sum_aggregator,
    ),
    'betas': MeasureConfig(
        loaders.load_betas,
        'betas',
    ),
    'beta_residuals': MeasureConfig(
        loaders.load_beta_residuals,
        'log_beta_residuals',
    ),
    'coefficients': MeasureConfig(
        loaders.load_coefficients,
        'coefficients',
    ),
    'scaling_parameters': MeasureConfig(
        loaders.load_scaling_parameters,
        'beta_scaling_parameters',
    ),
}


COMPOSITE_MEASURES = {
    'infection_fatality_ratio': CompositeMeasureConfig(
        base_measures={'infections': MEASURES['infections'],
                       'deaths': MEASURES['deaths']},
        label='infection_fatality_ratio',
        combiner=combiners.make_ifr,
    ),
}


COVARIATES = {
    'mobility': CovariateConfig(
        loaders.load_covariate,
        'mobility',
        time_varying=True,
        draw_level=True,
        aggregator=aggregators.mean_aggregator,
    ),
    'testing': CovariateConfig(
        loaders.load_covariate,
        'testing',
        time_varying=True,
        aggregator=aggregators.mean_aggregator,
    ),
    'pneumonia': CovariateConfig(
        loaders.load_covariate,
        'pneumonia',
        time_varying=True,
        aggregator=aggregators.mean_aggregator,
    ),
    'mask_use': CovariateConfig(
        loaders.load_covariate,
        'mask_use',
        time_varying=True,
        aggregator=aggregators.mean_aggregator,
    ),
    'air_pollution_pm_2_5': CovariateConfig(
        loaders.load_covariate,
        'air_pollution_pm_2_5',
        aggregator=aggregators.mean_aggregator,
    ),
    'lri_mortality': CovariateConfig(
        loaders.load_covariate,
        'lri_mortality',
        aggregator=aggregators.mean_aggregator,
    ),
    'proportion_over_2_5k': CovariateConfig(
        loaders.load_covariate,
        'proportion_over_2_5k',
        aggregator=aggregators.mean_aggregator,
    ),
    'proportion_under_100m': CovariateConfig(
        loaders.load_covariate,
        'proportion_under_100m',
        aggregator=aggregators.mean_aggregator,
    ),
    'smoking_prevalence': CovariateConfig(
        loaders.load_covariate,
        'smoking_prevalence',
        aggregator=aggregators.mean_aggregator,
    ),
}

MISCELLANEOUS = {
    'full_data': MiscellaneousConfig(
        loaders.load_full_data,
        'full_data',
        is_cumulative=True,
        aggregator=aggregators.sum_aggregator,
    ),
    'age_specific_deaths': MiscellaneousConfig(
        loaders.load_age_specific_deaths,
        'age_specific_deaths',
        aggregator=aggregators.sum_aggregator,
    ),
    'hospital_correction_factors': MiscellaneousConfig(
        loaders.load_hospital_correction_factors,
        'hospital_correction_factors',
    ),
    'hospital_census_data': MiscellaneousConfig(
        loaders.load_raw_census_data,
        'hospital_census_data',
    ),
    'version_map': MiscellaneousConfig(
        loaders.build_version_map,
        'version_map',
    ),
    'populations': MiscellaneousConfig(
        loaders.load_populations,
        'populations',
    ),
    'hierarchy': MiscellaneousConfig(
        loaders.load_hierarchy,
        'hierarchy',
    ),
    'locations_modeled_and_missing': MiscellaneousConfig(
        loaders.get_locations_modeled_and_missing,
        'modeled_and_missing_locations',
        is_table=False,
    ),
}
