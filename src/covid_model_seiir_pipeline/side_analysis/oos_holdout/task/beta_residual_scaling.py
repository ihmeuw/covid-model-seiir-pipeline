import click


from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline.pipeline.forecasting.task.beta_residual_scaling import (
    compute_initial_beta_scaling_parameters,
    write_out_beta_scale,
)
from covid_model_seiir_pipeline.side_analysis.oos_holdout.specification import (
    OOSHoldoutSpecification,
    OOS_HOLDOUT_JOBS,
)
from covid_model_seiir_pipeline.side_analysis.oos_holdout.data import (
    OOSHoldoutDataInterface,
)


logger = cli_tools.task_performance_logger


def run_oos_beta_scaling(oos_holdout_version: str, progress_bar: bool):
    logger.info(f"Computing beta scaling parameters for OOS version {oos_holdout_version}.", context='setup')

    specification = OOSHoldoutSpecification.from_version_root(oos_holdout_version)
    num_cores = specification.workflow.task_specifications[OOS_HOLDOUT_JOBS.oos_beta_scaling].num_cores
    data_interface = OOSHoldoutDataInterface.from_specification(specification)

    forecast_specification = data_interface.forecast_data_interface.load_specification()
    beta_scaling_parameters = forecast_specification.scenarios['reference'].beta_scaling

    logger.info('Computing scaling parameters.', context='compute')
    scaling_data = compute_initial_beta_scaling_parameters(
        beta_scaling_parameters,
        data_interface,
        num_cores,
        progress_bar
    )

    logger.info('Writing scaling parameters to disk.', context='write')
    write_out_beta_scale(scaling_data, 'reference', data_interface, num_cores)

    logger.report()


@click.command()
@cli_tools.with_task_oos_holdout_version
@cli_tools.with_progress_bar
@cli_tools.add_verbose_and_with_debugger
def oos_beta_scaling(oos_holdout_version: str,
                     progress_bar: bool,
                     verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_oos_beta_scaling, logger, with_debugger)
    run(oos_holdout_version=oos_holdout_version,
        progress_bar=progress_bar)
