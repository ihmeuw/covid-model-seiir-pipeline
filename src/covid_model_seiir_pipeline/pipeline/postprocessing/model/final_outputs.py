import itertools
from typing import Any, Callable, Dict, List, Union, TYPE_CHECKING

import pandas as pd

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    RISK_GROUP_NAMES,
    VARIANT_NAMES,
    VACCINE_STATUS_NAMES,
)
from covid_model_seiir_pipeline.pipeline.postprocessing.model import loaders, combiners
from covid_model_seiir_pipeline.lib import aggregate

if TYPE_CHECKING:
    # The model subpackage is a library for the pipeline stage and shouldn't
    # explicitly depend on things outside the subpackage.
    from covid_model_seiir_pipeline.pipeline.postprocessing.data import PostprocessingDataInterface


class MeasureConfig:
    def __init__(self,
                 loader: Callable[[str, 'PostprocessingDataInterface', int], Any],
                 label: str,
                 splice: bool = True,
                 resample: bool = True,
                 calculate_cumulative: bool = False,
                 cumulative_label: str = None,
                 aggregator: Callable = None,
                 write_draws: bool = False):
        self.loader = loader
        self.label = label
        self.splice = splice
        self.resample = resample
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
                 aggregator: Callable = None,
                 soft_fail: bool = False):
        self.loader = loader
        self.label = label
        self.is_table = is_table
        self.aggregator = aggregator
        self.soft_fail = soft_fail


DataConfig = Union[MeasureConfig, CovariateConfig, CompositeMeasureConfig, MiscellaneousConfig]


MEASURES = {}


for measure in ['infections', 'deaths', 'cases', 'admissions']:
    for suffix in list(VARIANT_NAMES[1:]) + list(RISK_GROUP_NAMES) + list(VACCINE_STATUS_NAMES) + ['total', 'naive', 'naive_unvaccinated']:
        measure_suffix = f'_{suffix}'
        label_suffix = f'_{suffix}' if suffix != 'total' else ''
        write_draws = suffix == 'total'
        label = f'{measure}{label_suffix}'
        MEASURES[label] = MeasureConfig(
            loaders.load_output_data(f'modeled_{measure}{measure_suffix}'),
            label=f'daily_{label}',
            calculate_cumulative=True,
            cumulative_label=f'cumulative_{label}',
            aggregator=aggregate.sum_aggregator,
            write_draws=write_draws,
            splice=suffix not in VACCINE_STATUS_NAMES,
        )
MEASURES.update(**{
    'unscaled_deaths': MeasureConfig(
        loaders.load_unscaled_deaths,
        'unscaled_daily_deaths',
        calculate_cumulative=True,
        cumulative_label='cumulative_unscaled_deaths',
        aggregator=aggregate.sum_aggregator,
        write_draws=True,
    ),
    'total_covid_deaths_data': MeasureConfig(
        loaders.load_total_covid_deaths,
        'total_covid_deaths_data',
        aggregator=aggregate.sum_aggregator,
    ),
    # Vaccination measures
    'cumulative_all_effective': MeasureConfig(
       loaders.load_vaccine_summaries('cumulative_all_effective'),
       'cumulative_vaccinations_all_effective',
       aggregator=aggregate.sum_aggregator,
       resample=False,
       splice=False,
    ),
    'cumulative_all_vaccinated': MeasureConfig(
       loaders.load_vaccine_summaries('cumulative_all_vaccinated'),
       'cumulative_vaccinations_all_vaccinated',
       aggregator=aggregate.sum_aggregator,
       resample=False,
       splice=False,
    ),
    'cumulative_all_fully_vaccinated': MeasureConfig(
       loaders.load_vaccine_summaries('cumulative_all_fully_vaccinated'),
       'cumulative_vaccinations_all_fully_vaccinated',
       aggregator=aggregate.sum_aggregator,
       resample=False,
       splice=False,
    ),
    'cumulative_lr_vaccinated': MeasureConfig(
       loaders.load_vaccine_summaries('lr_vaccinated'),
       'cumulative_vaccinations_lr',
       aggregator=aggregate.sum_aggregator,
       resample=False,
       splice=False,
    ),
    'cumulative_hr_vaccinated': MeasureConfig(
       loaders.load_vaccine_summaries('hr_vaccinated'),
       'cumulative_vaccinations_hr',
       aggregator=aggregate.sum_aggregator,
       resample=False,
       splice=False,
    ),
    'vaccine_acceptance': MeasureConfig(
       loaders.load_vaccine_summaries('vaccine_acceptance'),
       'vaccine_acceptance',
       aggregator=aggregate.mean_aggregator,
       resample=False,
       splice=False,
    ),
    'vaccine_acceptance_point': MeasureConfig(
       loaders.load_vaccine_summaries('vaccine_acceptance_point'),
       'vaccine_acceptance_point',
       aggregator=aggregate.mean_aggregator,
       resample=False,
       splice=False,
    ),
    # Betas
    'beta': MeasureConfig(
         loaders.load_ode_params('beta'),
         'betas',
    ),
    'beta_hat': MeasureConfig(
         loaders.load_ode_params('beta_hat'),
         'beta_hat',
    ),
    'empirical_beta': MeasureConfig(
        loaders.load_output_data('beta'),
        'empirical_beta',
    ),

    # Beta calculation inputs

    'beta_residuals': MeasureConfig(
        loaders.load_beta_residuals,
        'log_beta_residuals',
    ),
    'scaled_beta_residuals': MeasureConfig(
        loaders.load_scaled_beta_residuals,
        'scaled_log_beta_residuals',
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
    'infection_to_death': MeasureConfig(
        loaders.load_ode_params('exposure_to_death'),
        'infection_to_death',
    ),
    'infection_to_case': MeasureConfig(
        loaders.load_ode_params('exposure_to_case'),
        'infection_to_case',
    ),
    'infection_to_admission': MeasureConfig(
        loaders.load_ode_params('exposure_to_admission'),
        'infection_to_admission',
    ),
    'icu_admissions': MeasureConfig(
       loaders.load_output_data('icu_admissions'),
       'icu_admissions',
       aggregator=aggregate.sum_aggregator,
       write_draws=True,
    ),
    'hospital_census': MeasureConfig(
       loaders.load_output_data('hospital_census'),
       'hospital_census',
       aggregator=aggregate.sum_aggregator,
    ),
    'icu_census': MeasureConfig(
       loaders.load_output_data('icu_census'),
       'icu_census',
       aggregator=aggregate.sum_aggregator,
    ),
    'hospital_census_correction_factor': MeasureConfig(
       loaders.load_output_data('hospital_census_correction_factor'),
       'hospital_census_correction_factor',
    ),
    'icu_census_correction_factor': MeasureConfig(
       loaders.load_output_data('icu_census_correction_factor'),
       'icu_census_correction_factor',
    ),
})

for measure in ['boosters', 'vaccinations']:
    MEASURES[measure] = MeasureConfig(
        loaders.load_output_data(measure),
        f'daily_{measure}',
        calculate_cumulative=True,
        cumulative_label=f'cumulative_{measure}',
        aggregator=aggregate.sum_aggregator,
    )
    for risk_group in RISK_GROUP_NAMES:
        group_measure = f'{measure}_{risk_group}'
        MEASURES[group_measure] = MeasureConfig(
            loaders.load_output_data(group_measure),
            f'daily_{group_measure}',
            aggregator=aggregate.sum_aggregator,
        )


for key in list(VARIANT_NAMES[1:]) + list(RISK_GROUP_NAMES) + ['total']:
    MEASURES[f'susceptible_{key}'] = MeasureConfig(
        loaders.load_output_data(f'susceptible_{key}'),
        f'effective_susceptible_{key}',
        aggregator=aggregate.sum_aggregator,
    )


for covid_exposure, vaccine_status in itertools.product(['naive', 'exposed'], ['unvaccinated', 'vaccinated']):
    MEASURES[f'covid_status_{covid_exposure}_{vaccine_status}'] = MeasureConfig(
        loaders.load_output_data(f'covid_status_{covid_exposure}_{vaccine_status}'),
        f'covid_status_{covid_exposure}_{vaccine_status}',
        aggregator=aggregate.sum_aggregator,
    )


for key in list(VARIANT_NAMES[1:]) + ['total']:
    measure = f'force_of_infection_{key}'
    MEASURES[measure] = MeasureConfig(
        loaders.load_output_data(measure),
        measure,
    )

for variant in VARIANT_NAMES[1:]:
    MEASURES[f'immune_{variant}'] = MeasureConfig(
        loaders.load_output_data(f'immune_{variant}'),
        f'effective_immune_{variant}',
        aggregator=aggregate.sum_aggregator,
    )
    MEASURES[f'variant_{variant}_prevalence'] = MeasureConfig(
        loaders.load_output_data(f'variant_{variant}_prevalence'),
        f'variant_prevalence_{variant}',
    )


for key in list(VARIANT_NAMES[1:]) + ['total']:
    for measure in ['r_effective', 'r_controlled']:
        group_measure = f'{measure}_{key}'
        MEASURES[group_measure] = MeasureConfig(
            loaders.load_output_data(group_measure),
            group_measure,
        )


COMPOSITE_MEASURES = {}
ratio_map = [('infection_fatality_ratio', 'deaths', 'infection_to_death'),
             ('infection_hospitalization_ratio', 'admissions', 'infection_to_admission'),
             ('infection_detection_ratio', 'cases', 'infection_to_case')]
for ratio, measure, lag in ratio_map:
    for risk_group in ['', '_high_risk', '_low_risk']:
        risk_group_short = '_' + ''.join([s[0] for s in risk_group[1:].split('_')]) if risk_group else ''

        COMPOSITE_MEASURES[f'{ratio}{risk_group}'] = CompositeMeasureConfig(
            base_measures={'numerator': MEASURES[f'{measure}{risk_group_short}'],
                           'denominator': MEASURES[f'infections{risk_group_short}'],
                           'duration': MEASURES[lag]},
            label=f'{ratio}{risk_group}',
            combiner=combiners.make_ratio,
        )

COVARIATES = {
    'mobility': CovariateConfig(
        loaders.load_covariate,
        'mobility',
        time_varying=True,
        aggregator=aggregate.mean_aggregator,
    ),
    'testing': CovariateConfig(
        loaders.load_covariate,
        'testing',
        time_varying=True,
        aggregator=aggregate.mean_aggregator,
    ),
    'pneumonia': CovariateConfig(
        loaders.load_covariate,
        'pneumonia',
        time_varying=True,
        aggregator=aggregate.mean_aggregator,
    ),
    'mask_use': CovariateConfig(
        loaders.load_covariate,
        'mask_use',
        time_varying=True,
        aggregator=aggregate.mean_aggregator,
    ),
    'air_pollution_pm_2_5': CovariateConfig(
        loaders.load_covariate,
        'air_pollution_pm_2_5',
        aggregator=aggregate.mean_aggregator,
    ),
    'lri_mortality': CovariateConfig(
        loaders.load_covariate,
        'lri_mortality',
        aggregator=aggregate.mean_aggregator,
    ),
    'proportion_over_2_5k': CovariateConfig(
        loaders.load_covariate,
        'proportion_over_2_5k',
        aggregator=aggregate.mean_aggregator,
    ),
    'proportion_under_100m': CovariateConfig(
        loaders.load_covariate,
        'proportion_under_100m',
        aggregator=aggregate.mean_aggregator,
    ),
    'smoking_prevalence': CovariateConfig(
        loaders.load_covariate,
        'smoking_prevalence',
        aggregator=aggregate.mean_aggregator,
    ),
}

MISCELLANEOUS = {
    'unscaled_full_data': MiscellaneousConfig(
        loaders.load_full_data_unscaled,
        'unscaled_full_data',
        aggregator=aggregate.sum_aggregator,
    ),
    'variant_prevalence': MiscellaneousConfig(
        loaders.load_variant_prevalence,
        'variant_prevalence',
        aggregator=aggregate.mean_aggregator,
    ),
    'excess_mortality_scalars': MiscellaneousConfig(
        loaders.load_excess_mortality_scalars,
        'excess_mortality_scalars',
    ),
    'hospital_census_data': MiscellaneousConfig(
        loaders.load_raw_census_data,
        'hospital_census_data',
    ),
    'hospital_bed_capacity': MiscellaneousConfig(
        loaders.load_hospital_bed_capacity,
        'hospital_bed_capacity',
    ),
    'vaccine_efficacy_table': MiscellaneousConfig(
        loaders.load_vaccine_efficacy_table,
        'vaccine_efficacy_table',
    ),
    'populations': MiscellaneousConfig(
        loaders.load_populations,
        'populations',
        aggregator=aggregate.sum_aggregator,
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
