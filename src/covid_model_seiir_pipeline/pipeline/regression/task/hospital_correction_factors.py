from dataclasses import asdict
from pathlib import Path

import click
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    static_vars,
    math,
)
from covid_model_seiir_pipeline.pipeline.regression.specification import (
    RegressionSpecification,
)
from covid_model_seiir_pipeline.pipeline.regression.data import (
    RegressionDataInterface,
)
from covid_model_seiir_pipeline.pipeline.regression import (
    model,
)


logger = cli_tools.task_performance_logger


def run_hospital_correction_factors(regression_version: str, with_error: bool) -> None:
    logger.info('Starting hospital correction factors.', context='setup')
    # Build helper abstractions
    regression_spec_file = Path(regression_version) / static_vars.REGRESSION_SPECIFICATION_FILE
    regression_specification = RegressionSpecification.from_path(regression_spec_file)
    hospital_parameters = regression_specification.hospital_parameters
    data_interface = RegressionDataInterface.from_specification(regression_specification)

    logger.info('Loading input data', context='read')
    hierarchy = data_interface.load_hierarchy()
    location_ids = data_interface.load_location_ids()
    # We just want the mean deaths through here, which is the same across
    # all draws, so we'll default to draw 0.
    infection_data = data_interface.load_past_infection_data(
        draw_id=0,
        location_ids=location_ids,
    )
    deaths = infection_data['deaths'].reset_index()

    population = data_interface.load_five_year_population(location_ids)
    mr = data_interface.load_mortality_ratio(location_ids)
    death_weights = model.get_death_weights(mr, population, with_error)
    hfr = data_interface.load_hospital_fatality_ratio(death_weights, location_ids, with_error)
    hospital_census_data = data_interface.load_hospital_census_data()

    logger.info('Computing hospital usage', context='compute_usage')
    hospital_usage = model.compute_hospital_usage(
        deaths,
        death_weights,
        hfr,
        hospital_parameters,
    )
    logger.info('Computing correction factors', context='compute_corrections')
    correction_factors = model.calculate_hospital_correction_factors(
        hospital_usage,
        hospital_census_data,
        hierarchy,
        hospital_parameters,
    )

    logger.info('Prepping outputs', context='transform')
    usage = [value.rename(key) for key, value in asdict(hospital_usage).items()]
    usage_df = pd.concat(usage, axis=1).reset_index()
    corrections = [value.rename(key) for key, value in asdict(correction_factors).items()]
    corrections_df = pd.concat(corrections, axis=1).reset_index()

    logger.info('Writing outputs', context='write')
    data_interface.save_hospital_data(usage_df, 'usage')
    data_interface.save_hospital_data(corrections_df, 'correction_factors')

    logger.report()


@click.command()
@cli_tools.with_regression_version
@click.option('-e', 'with_error', is_flag=True)
@cli_tools.add_verbose_and_with_debugger
def hospital_correction_factors(regression_version: str,
                                with_error: bool, verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)

    run = cli_tools.handle_exceptions(run_hospital_correction_factors, logger, with_debugger)
    run(regression_version=regression_version, with_error=with_error)


if __name__ == '__main__':
    hospital_correction_factors()
