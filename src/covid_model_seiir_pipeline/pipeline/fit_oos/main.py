from covid_shared import cli_tools
from loguru import logger

from covid_model_seiir_pipeline.lib.ihme_deps import WorkflowAlreadyComplete
from covid_model_seiir_pipeline.pipeline.fit_oos.specification import FitSpecification
from covid_model_seiir_pipeline.pipeline.fit_oos.workflow import FitWorkflow
from covid_model_seiir_pipeline.pipeline.forecasting.data import ForecastDataInterface



def do_beta_fit(app_metadata: cli_tools.Metadata,
                forecast_specification: ForecastSpecification,
                preprocess_only: bool):
    logger.info(f'Starting beta forecast for version {forecast_specification.data.output_root}.')

    data_interface = ForecastDataInterface.from_specification(forecast_specification)
    if not data_interface.has_hospital_corrections():
        raise ValueError(
            'No hospital correction factors were produced in the regression stage. '
            'Forecast model cannot be run using this regression version.'
        )

    # Check scenario covariates the same as regression covariates and that
    # covariate data versions match.
    data_interface.check_covariates(forecast_specification.scenarios)

    data_interface.make_dirs(scenario=list(forecast_specification.scenarios))
    data_interface.save_specification(forecast_specification)

    if not preprocess_only:
        forecast_wf = ForecastWorkflow(forecast_specification.data.output_root,
                                       forecast_specification.workflow)
        n_draws = data_interface.get_n_draws()

        forecast_wf.attach_tasks(n_draws=n_draws,
                                 scenarios=forecast_specification.scenarios)
        try:
            forecast_wf.run()
        except WorkflowAlreadyComplete:
            logger.info('Workflow already complete')
