from covid_shared import cli_tools
from jobmon.client.swarm.workflow.workflow import WorkflowAlreadyComplete
from loguru import logger

from covid_model_seiir_pipeline.postprocessing.data import PostprocessingDataInterface
from covid_model_seiir_pipeline.postprocessing.specification import PostprocessingSpecification
from covid_model_seiir_pipeline.postprocessing.workflow import PostprocessingWorkflow


def do_postprocessing(app_metadata: cli_tools.Metadata,
                      postprocessing_specification: PostprocessingSpecification,
                      preprocess_only: bool):
    logger.info(f'Starting postprocessing for version {postprocessing_specification.data.output_root}.')

    data_interface = PostprocessingDataInterface.from_specification(postprocessing_specification)

    data_interface.make_dirs(scenario=postprocessing_specification.data.scenarios)
    data_interface.save_specification(postprocessing_specification)

    if not preprocess_only:
        workflow = PostprocessingWorkflow(postprocessing_specification.data.output_root,
                                          postprocessing_specification.workflow)
        covariates = data_interface.get_covariates(postprocessing_specification.data.scenarios)
        workflow.attach_tasks(postprocessing_specification.data.scenarios, covariates)

        try:
            workflow.run()
        except WorkflowAlreadyComplete:
            logger.info('Workflow already complete')
