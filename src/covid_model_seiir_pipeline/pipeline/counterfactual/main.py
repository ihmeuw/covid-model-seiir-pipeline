from covid_shared import cli_tools, ihme_deps
from loguru import logger

from covid_model_seiir_pipeline.pipeline.counterfactual.specification import CounterfactualSpecification
from covid_model_seiir_pipeline.pipeline.counterfactual.data import CounterfactualDataInterface
from covid_model_seiir_pipeline.pipeline.counterfactual.workflow import CounterfactualWorkflow


def do_counterfactual(app_metadata: cli_tools.Metadata,
                      counterfactual_specification: CounterfactualSpecification,
                      preprocess_only: bool):
    logger.info(f'Starting counterfactual for version {counterfactual_specification.data.output_root}.')

    data_interface = CounterfactualDataInterface.from_specification(counterfactual_specification)

    data_interface.make_dirs(scenario=list(counterfactual_specification.scenarios))
    data_interface.save_specification(counterfactual_specification)

    if not preprocess_only:
        counterfactual_wf = CounterfactualWorkflow(
            counterfactual_specification.data.output_root,
            counterfactual_specification.workflow,
        )
        n_draws = data_interface.get_n_draws()
        counterfactual_wf.attach_tasks(
            n_draws=n_draws,
            scenarios=counterfactual_specification.scenarios
        )
        try:
            counterfactual_wf.run()
        except ihme_deps.WorkflowAlreadyComplete:
            logger.info('Workflow already complete')
