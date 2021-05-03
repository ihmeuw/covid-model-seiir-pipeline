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
                 aggregator: Callable = None,
                 write_draws: bool = False):
        self.loader = loader
        self.label = label
        self.splice = splice
        self.calculate_cumulative = calculate_cumulative
        self.cumulative_label = cumulative_label
        self.aggregator = aggregator
        self.write_draws = write_draws


class CompositeMeasureConfig:
    def __init__(self,
                 base_measures: Dict[str, MeasureConfig],
                 label: str,
                 combiner: Callable,
                 write_draws: bool = False):
        self.base_measures = base_measures
        self.label = label
        self.combiner = combiner
        self.write_draws = write_draws


class CovariateConfig:
    def __init__(self,
                 loader: Callable[[str, bool, str, 'PostprocessingDataInterface', int], List[pd.Series]],
                 label: str,
                 splice: bool = False,
                 time_varying: bool = False,
                 aggregator: Callable = None,
                 write_draws: bool = False):
        self.loader = loader
        self.label = label
        self.splice = splice
        self.time_varying = time_varying
        self.aggregator = aggregator
        self.write_draws = write_draws


class MiscellaneousConfig:
    def __init__(self,
                 loader: Callable[['PostprocessingDataInterface'], Any],
                 label: str,
                 is_table: bool = True,
                 aggregator: Callable = None):
        self.loader = loader
        self.label = label
        self.is_table = is_table
        self.aggregator = aggregator


MEASURES = {
    # Death measures

    'deaths': MeasureConfig(
        loaders.load_deaths,
        'daily_deaths',
        calculate_cumulative=True,
        cumulative_label='cumulative_deaths',
        aggregator=aggregators.sum_aggregator,
        write_draws=True,
    ),
    'unscaled_deaths': MeasureConfig(
        loaders.load_unscaled_deaths,
        'unscaled_daily_deaths',
        calculate_cumulative=True,
        cumulative_label='cumulative_unscaled_deaths',
        aggregator=aggregators.sum_aggregator,
        write_draws=True,
    ),
    'deaths_wild': MeasureConfig(
        loaders.load_deaths_wild,
        'daily_deaths_wild',
        aggregator=aggregators.sum_aggregator,
    ),
    'deaths_variant': MeasureConfig(
        loaders.load_deaths_variant,
        'daily_deaths_variant',
        aggregator=aggregators.sum_aggregator,
    ),

    # Infection measures

    'infections': MeasureConfig(
        loaders.load_infections,
        'daily_infections',
        calculate_cumulative=True,
        cumulative_label='cumulative_infections',
        aggregator=aggregators.sum_aggregator,
        write_draws=True,
    ),
    'infections_wild': MeasureConfig(
        loaders.load_infections_wild,
        'daily_infections_wild',
        aggregator=aggregators.sum_aggregator,
    ),
    'infections_variant': MeasureConfig(
        loaders.load_infections_variant,
        'daily_infections_variant',
        aggregator=aggregators.sum_aggregator,
    ),
    'infections_natural_breakthrough': MeasureConfig(
        loaders.load_infections_natural_breakthrough,
        'daily_infections_natural_immunity_breakthrough',
        aggregator=aggregators.sum_aggregator,
    ),
    'infections_vaccine_breakthrough': MeasureConfig(
        loaders.load_infections_vaccine_breakthrough,
        'daily_infections_vaccine_breakthrough',
        aggregator=aggregators.sum_aggregator,
    ),
    'cases': MeasureConfig(
        loaders.load_cases,
        'daily_cases',
        calculate_cumulative=True,
        cumulative_label='cumulative_cases',
        aggregator=aggregators.sum_aggregator,
    ),

    # Hospital measures

    'hospital_admissions': MeasureConfig(
        loaders.load_hospital_admissions,
        'hospital_admissions',
        aggregator=aggregators.sum_aggregator,
        write_draws=True,
    ),
    'icu_admissions': MeasureConfig(
        loaders.load_icu_admissions,
        'icu_admissions',
        aggregator=aggregators.sum_aggregator,
        write_draws=True,
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

    # Vaccination measures

    'effectively_vaccinated': MeasureConfig(
        loaders.load_effectively_vaccinated,
        'daily_vaccinations_effective_input',
        calculate_cumulative=True,
        cumulative_label='cumulative_vaccinations_effective_input',
        aggregator=aggregators.sum_aggregator,
    ),
    'vaccines_immune_all': MeasureConfig(
        loaders.load_vaccinations_immune_all,
        'daily_vaccinations_all_immune',
        aggregator=aggregators.sum_aggregator,
    ),
    'vaccines_immune_wild': MeasureConfig(
        loaders.load_vaccinations_immune_wild,
        'daily_vaccinations_wild_immune',
        aggregator=aggregators.sum_aggregator,
    ),
    'vaccines_protected_all': MeasureConfig(
        loaders.load_vaccinations_protected_all,
        'daily_vaccinations_all_protected',
        aggregator=aggregators.sum_aggregator,
    ),
    'vaccines_protected_wild': MeasureConfig(
        loaders.load_vaccinations_protected_wild,
        'daily_vaccinations_wild_protected',
        aggregator=aggregators.sum_aggregator,
    ),
    'vaccines_effective': MeasureConfig(
        loaders.load_vaccinations_effective,
        'daily_vaccinations_effective',
        calculate_cumulative=True,
        cumulative_label='cumulative_vaccinations_effective',
        aggregator=aggregators.sum_aggregator,
    ),
    'vaccines_ineffective': MeasureConfig(
        loaders.load_vaccinations_ineffective,
        'daily_vaccinations_ineffective',
        aggregator=aggregators.sum_aggregator,
    ),

    # Other epi measures

    'total_susceptible_wild': MeasureConfig(
        loaders.load_total_susceptible_wild,
        'total_susceptible_wild',
        aggregator=aggregators.sum_aggregator,
    ),
    'total_susceptible_variant': MeasureConfig(
        loaders.load_total_susceptible_variant,
        'total_susceptible_variant',
        aggregator=aggregators.sum_aggregator,
    ),
    'total_susceptible_variant_only': MeasureConfig(
        loaders.load_total_susceptible_variant_only,
        'total_susceptible_variant_only',
        aggregator=aggregators.sum_aggregator,
    ),
    'total_immune_wild': MeasureConfig(
        loaders.load_total_immune_wild,
        'total_immune_wild',
        aggregator=aggregators.sum_aggregator,
    ),
    'total_immune_variant': MeasureConfig(
        loaders.load_total_immune_variant,
        'total_immune_variant',
        aggregator=aggregators.sum_aggregator,
    ),
    'r_controlled_wild': MeasureConfig(
        loaders.load_r_controlled_wild,
        'r_controlled_wild',
    ),
    'r_effective_wild': MeasureConfig(
        loaders.load_r_effective_wild,
        'r_effective_wild',
    ),
    'r_controlled_variant': MeasureConfig(
        loaders.load_r_controlled_variant,
        'r_controlled_variant',
    ),
    'r_effective_variant': MeasureConfig(
        loaders.load_r_effective_variant,
        'r_effective_variant',
    ),
    'r_effective': MeasureConfig(
        loaders.load_r_effective,
        'r_effective',
    ),

    # Betas

    'beta': MeasureConfig(
        loaders.load_beta,
        'betas',
    ),
    'beta_hat': MeasureConfig(
        loaders.load_beta_hat,
        'beta_hat',
    ),
    'beta_wild': MeasureConfig(
        loaders.load_beta_wild,
        'beta_wild',
    ),
    'beta_variant': MeasureConfig(
        loaders.load_beta_variant,
        'beta_variant',
    ),
    'empirical_beta': MeasureConfig(
        loaders.load_empirical_beta,
        'empirical_beta',
    ),
    'empirical_beta_wild': MeasureConfig(
        loaders.load_empirical_beta_wild,
        'empirical_beta_wild',
    ),
    'empirical_beta_variant': MeasureConfig(
        loaders.load_empirical_beta_variant,
        'empirical_beta_variant',
    ),

    'non_escape_variant_prevalence': MeasureConfig(
        loaders.load_non_escape_variant_prevalence,
        'non_escape_variant_prevalence',
    ),
    'escape_variant_prevalence': MeasureConfig(
        loaders.load_escape_variant_prevalence,
        'escape_variant_prevalence',
    ),

    # Beta calculation inputs

    'beta_residuals': MeasureConfig(
        loaders.load_beta_residuals,
        'log_beta_residuals',
    ),
    'coefficients': MeasureConfig(
        loaders.load_coefficients,
        'coefficients',
        write_draws=True,
    ),
    'scaling_parameters': MeasureConfig(
        loaders.load_scaling_parameters,
        'beta_scaling_parameters',
        write_draws=True,
    ),
    'infection_fatality_ratio_es': MeasureConfig(
        loaders.load_ifr_es,
        'infection_fatality_ratio_es',
    ),
    'infection_detection_ratio_es': MeasureConfig(
        loaders.load_idr_es,
        'infection_detection_ratio_es',
    ),
    'infection_hospitalization_ratio_es': MeasureConfig(
        loaders.load_ihr_es,
        'infection_hospitalization_ratio_es',
    ),
}


COMPOSITE_MEASURES = {
    'infection_fatality_ratio': CompositeMeasureConfig(
        base_measures={'infections': MEASURES['infections'],
                       'deaths': MEASURES['deaths']},
        label='infection_fatality_ratio',
        combiner=combiners.make_ifr,
    ),
    'infection_hospitalization_ratio': CompositeMeasureConfig(
        base_measures={'infections': MEASURES['infections'],
                       'hospital_admissions': MEASURES['hospital_admissions']},
        label='infection_hospitalization_ratio',
        combiner=combiners.make_ihr,
    ),
    'infection_detection_ratio': CompositeMeasureConfig(
        base_measures={'infections': MEASURES['infections'],
                       'cases': MEASURES['cases']},
        label='infection_detection_ratio',
        combiner=combiners.make_idr,
    ),
    'empirical_escape_variant_prevalence': CompositeMeasureConfig(
        base_measures={'escape_variant_infections': MEASURES['infections_variant'],
                       'total_infections': MEASURES['infections']},
        label='empirical_escape_variant_prevalence',
        combiner=combiners.make_empirical_escape_variant_prevalence,
    ),
}


COVARIATES = {
    'mobility': CovariateConfig(
        loaders.load_covariate,
        'mobility',
        time_varying=True,
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
        aggregator=aggregators.sum_aggregator,
    ),
    'unscaled_full_data': MiscellaneousConfig(
        loaders.load_unscaled_full_data,
        'unscaled_full_data',
        aggregator=aggregators.sum_aggregator,
    ),
    'age_specific_deaths': MiscellaneousConfig(
        loaders.load_age_specific_deaths,
        'age_specific_deaths',
        aggregator=aggregators.sum_aggregator,
    ),
    'excess_mortality_scalars': MiscellaneousConfig(
        loaders.load_excess_mortality_scalars,
        'excess_mortality_scalars',
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
