from typing import Optional, Union

import click
from covid_shared import ihme_deps, paths
from loguru import logger

from covid_model_seiir_pipeline.lib import cli_tools

from covid_model_seiir_pipeline.pipeline.preprocessing.specification import PreprocessingSpecification
from covid_model_seiir_pipeline.pipeline.preprocessing.data import PreprocessingDataInterface
from covid_model_seiir_pipeline.pipeline.preprocessing.workflow import PreprocessingWorkflow


def do_preprocessing(*args, **kwargs) -> PreprocessingSpecification:
    pass


def preprocessing_main(app_metadata: cli_tools.Metadata,
                       preprocessing_specification: PreprocessingSpecification,
                       preprocess_only: bool):
    pass


@click.command()
@cli_tools.pass_run_metadata()
@cli_tools.with_specification(PreprocessingSpecification)
@cli_tools.with_location_specification
@cli_tools.with_model_inputs_version
@cli_tools.with_age_specific_rates_version
@cli_tools.with_mortality_scalars_version
@cli_tools.with_mask_use_version
@cli_tools.with_mobility_version
@cli_tools.with_pneumonia_version
@cli_tools.with_population_density_version
@cli_tools.with_testing_version
@cli_tools.with_variant_prevalence_version
@cli_tools.with_vaccine_coverage_version
@cli_tools.with_vaccine_efficacy_version
@cli_tools.add_preprocess_only
@cli_tools.add_output_options(paths.SEIR_PREPROCESS_ROOT)
@cli_tools.add_verbose_and_with_debugger
def preprocess(run_metadata: cli_tools.RunMetadata,
               preprocessing_specification: str,
               location_specification: Union[str, int],
               model_inputs_version: str,
               age_specific_rates_version: str,
               mortality_scalars_version: str,
               mask_use_version: str,
               mobility_version: str,
               pneumonia_version: str,
               population_density_version: str,
               testing_version: str,
               variant_prevalence_version: str,
               vaccine_coverage_version: str,
               vaccine_efficacy_version

):
    pass
