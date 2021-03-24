from covid_shared import cli_tools
from loguru import logger

from covid_model_seiir_pipeline.lib.ihme_deps import WorkflowAlreadyComplete
from covid_model_seiir_pipeline.pipeline.fit_oos.specification import FitSpecification
from covid_model_seiir_pipeline.pipeline.fit_oos.workflow import FitWorkflow
from covid_model_seiir_pipeline.pipeline.fit_oos.data import FitDataInterface


def do_beta_fit(app_metadata: cli_tools.Metadata,
                fit_specification: FitSpecification,
                preprocess_only: bool):
    logger.info(f'Starting beta fit for version {fit_specification.data.output_root}.')

    data_interface = FitDataInterface.from_specification(fit_specification)

    data_interface.make_dirs(scenario=list(fit_specification.scenarios))
    data_interface.save_specification(fit_specification)

    if not preprocess_only:
        fit_wf = FitWorkflow(fit_specification.data.output_root,
                             fit_specification.workflow)
        n_draws = data_interface.get_n_draws()

        fit_wf.attach_tasks(n_draws=n_draws,
                            scenarios=fit_specification.scenarios)
        try:
            fit_wf.run()
        except WorkflowAlreadyComplete:
            logger.info('Workflow already complete')
