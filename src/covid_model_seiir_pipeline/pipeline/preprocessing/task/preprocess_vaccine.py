import click
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.specification import (
    PreprocessingSpecification,
    PREPROCESSING_JOBS,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.data import (
    PreprocessingDataInterface,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.model import (
    vaccination as model,
)

logger = cli_tools.task_performance_logger


def run_preprocess_vaccine(preprocessing_version: str, scenario: str, progress_bar: bool) -> None:
    logger.info(f'Starting vaccination preprocessing for scenario {scenario}.', context='setup')

    specification = PreprocessingSpecification.from_version_root(preprocessing_version)
    data_interface = PreprocessingDataInterface.from_specification(specification)

    logger.info(f'Loading efficacy and waning data.', context='read')
    efficacy = data_interface.load_efficacy_table()
    waning = data_interface.load_vaccine_waning_distribution()
    summary = data_interface.load_serology_vaccine_coverage()

    logger.info('Prepping data for waning efficacy calculations', context='transform')
    efficacy = model.map_variants(efficacy)
    all_brands = efficacy.reset_index().brand.unique().tolist()

    waning = model.rescale_to_proportion(waning)
    infection_waning = model.get_infection_endpoint_brand_specific_waning(waning, all_brands)
    severe_disease_waning = model.get_severe_endpoint_brand_specific_waning(waning, all_brands)
    waning = pd.concat([infection_waning, severe_disease_waning])

    logger.info('Computing brand and variant specific waning efficacy.', context='model')
    waning_efficacy = model.build_waning_efficacy(efficacy, waning)
    logger.info('Computing natural waning distribution.', context='model')
    natural_waning = model.compute_natural_waning(waning)
    if scenario == 'base_measures':
        logger.info('Writing base measure results.', context='write')
        data_interface.save_waning_parameters(efficacy, 'base_vaccine_efficacy')
        data_interface.save_waning_parameters(waning, 'vaccine_waning_distribution')
        data_interface.save_waning_parameters(waning_efficacy, 'vaccine_waning_efficacy')
        data_interface.save_waning_parameters(natural_waning, 'natural_waning_distribution')
        data_interface.save_vaccine_summary(summary)
    else:
        logger.info(f'Loading uptake data for scenario {scenario}.', context='read')
        uptake = data_interface.load_raw_vaccine_uptake(scenario)
        logger.info(f'Broadcasting uptake data over shared index.', context='transform')
        uptake = model.make_uptake_square(uptake)

        logger.info('Building vaccine risk reduction argument list.', context='model')
        eta_args = model.build_eta_calc_arguments(uptake, waning_efficacy, progress_bar)
        logger.info('Computing vaccine risk reductions.', context='model')
        num_cores = specification.workflow.task_specifications[PREPROCESSING_JOBS.preprocess_vaccine].num_cores
        risk_reductions = model.build_vaccine_risk_reduction(eta_args, num_cores, progress_bar)
        uptake = (uptake
                  .reorder_levels(['location_id', 'date', 'vaccine_course', 'risk_group'])
                  .sort_index()
                  .sum(axis=1)
                  .unstack()
                  .unstack())
        uptake.columns = [f'course_{course}_{risk_group}' for risk_group, course in uptake]

        logger.info(f'Writing uptake and risk reductions for scenario {scenario}.', context='write')
        data_interface.save_vaccine_uptake(uptake, scenario=scenario)
        data_interface.save_vaccine_risk_reduction(risk_reductions, scenario=scenario)

    logger.report()


@click.command()
@cli_tools.with_task_preprocessing_version
@cli_tools.with_scenario
@cli_tools.add_verbose_and_with_debugger
@cli_tools.with_progress_bar
def preprocess_vaccine(preprocessing_version: str, scenario: str,
                       verbose: int, with_debugger: bool, progress_bar: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_preprocess_vaccine, logger, with_debugger)
    run(preprocessing_version=preprocessing_version,
        scenario=scenario,
        progress_bar=progress_bar)
