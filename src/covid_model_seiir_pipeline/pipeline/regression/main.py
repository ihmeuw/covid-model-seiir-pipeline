from covid_shared import cli_tools
from loguru import logger

from covid_model_seiir_pipeline.lib.ihme_deps import WorkflowAlreadyComplete
from covid_model_seiir_pipeline.pipeline.regression.specification import RegressionSpecification
from covid_model_seiir_pipeline.pipeline.regression.data import RegressionDataInterface
from covid_model_seiir_pipeline.pipeline.regression.workflow import RegressionWorkflow


def do_beta_regression(app_metadata: cli_tools.Metadata,
                       regression_specification: RegressionSpecification,
                       preprocess_only: bool):
    logger.info(f'Starting beta regression for version {regression_specification.data.output_root}.')

    # init high level objects
    data_interface = RegressionDataInterface.from_specification(regression_specification)

    # Grab canonical location list from arguments
    hierarchy = data_interface.load_hierarchy_from_primary_source(
        location_set_version_id=regression_specification.data.location_set_version_id,
        location_file=regression_specification.data.location_set_file
    )
    # Filter to the intersection of what's available from the infection data.
    location_ids = data_interface.filter_location_ids(hierarchy)

    # Check to make sure we have all the covariates we need
    data_interface.check_covariates(regression_specification.covariates)

    # build directory structure and save metadata
    data_interface.make_dirs()
    data_interface.save_location_ids(location_ids)
    data_interface.save_specification(regression_specification)
    data_interface.save_hierarchy(hierarchy)

    # build workflow and launch
    if not preprocess_only:
        regression_wf = RegressionWorkflow(regression_specification.data.output_root,
                                           regression_specification.workflow)
        regression_wf.attach_tasks(n_draws=data_interface.get_n_draws())
        try:
            regression_wf.run()
        except WorkflowAlreadyComplete:
            logger.info('Workflow already complete.')
