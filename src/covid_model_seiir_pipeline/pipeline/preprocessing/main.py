from covid_shared import cli_tools, ihme_deps
from loguru import logger

from covid_model_seiir_pipeline.pipeline.preprocessing.specification import PreprocessingSpecification
from covid_model_seiir_pipeline.pipeline.preprocessing.data import PreprocessingDataInterface
from covid_model_seiir_pipeline.pipeline.preprocessing.workflow import PreprocessingWorkflow


def do_beta_regression(app_metadata: cli_tools.Metadata,
                       preprocessing_specification: PreprocessingSpecification,
                       preprocess_only: bool):
    logger.info(f'Starting preprocessing for version {preprocessing_specification.data.output_root}.')

    data_interface = PreprocessingDataInterface.from_specification(preprocessing_specification)
    # build directory structure and save metadata
    data_interface.make_dirs()
    data_interface.save_specification(preprocessing_specification)

    # Grab canonical location list from arguments
    hierarchy = data_interface.load_hierarchy_from_primary_source(
        location_set_version_id=preprocessing_specification.data.location_set_version_id,
        location_file=preprocessing_specification.data.location_set_file
    )
    data_interface.save_modeling_hierarchy(hierarchy)

    # build workflow and launch
    if not preprocess_only:
        regression_wf = PreprocessingWorkflow(preprocessing_specification.data.output_root,
                                              preprocessing_specification.workflow)
        regression_wf.attach_tasks(n_draws=data_interface.get_n_draws())
        try:
            regression_wf.run()
        except ihme_deps.WorkflowAlreadyComplete:
            logger.info('Workflow already complete.')
