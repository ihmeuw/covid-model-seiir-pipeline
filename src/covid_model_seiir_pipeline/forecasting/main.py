from pathlib import Path

from covid_shared import cli_tools
from jobmon.client.swarm.workflow.workflow import WorkflowAlreadyComplete
from loguru import logger

from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification, PostprocessingSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.forecasting.workflow import ForecastWorkflow, PostprocessingWorkflow


def do_beta_forecast(app_metadata: cli_tools.Metadata,
                     forecast_specification: ForecastSpecification,
                     preprocess_only: bool):
    logger.debug('Starting beta forecast.')

    data_interface = ForecastDataInterface.from_specification(forecast_specification)

    # Check scenario covariates the same as regression covariates and that
    # covariate data versions match.
    data_interface.check_covariates(forecast_specification.scenarios)

    data_interface.make_dirs()
    # Fixme: Inconsistent data writing interfaces
    forecast_specification.dump(data_interface.forecast_paths.forecast_specification)

    if not preprocess_only:
        forecast_wf = ForecastWorkflow(forecast_specification.data.output_root)
        n_draws = data_interface.get_n_draws()

        forecast_wf.attach_tasks(n_draws=n_draws,
                                 scenarios=forecast_specification.scenarios)
        try:
            forecast_wf.run()
        except WorkflowAlreadyComplete:
            logger.info('Workflow already complete')


def do_postprocessing(app_metadata: cli_tools.Metadata,
                      postprocessing_specification: PostprocessingSpecification,
                      preprocess_only: bool):
    logger.debug('Starting postprocessing')
    forecast_specification = ForecastSpecification.from_path(
        Path(postprocessing_specification.data.forecast_version) / 'forecast_specification.yaml'
    )
    data_interface = ForecastDataInterface.from_specification(forecast_specification, postprocessing_specification)

    # Check scenario covariates the same as regression covariates and that
    # covariate data versions match.
    covariates = data_interface.check_covariates(forecast_specification.scenarios)

    data_interface.make_dirs()
    postprocessing_specification.dump(data_interface.postprocessing_paths.postprocessing_specification)

    if not preprocess_only:
        workflow = PostprocessingWorkflow(postprocessing_specification.data.output_root)
        workflow.attach_tasks(forecast_specification.scenarios, covariates)

        try:
            workflow.run()
        except WorkflowAlreadyComplete:
            logger.info('Workflow already complete')
