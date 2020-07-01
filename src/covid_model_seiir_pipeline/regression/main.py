from covid_shared import cli_tools
from loguru import logger

from covid_model_seiir_pipeline.regression.specification import RegressionSpecification
from covid_model_seiir_pipeline.regression.data import RegressionDataInterface
from covid_model_seiir_pipeline.regression.workflow import RegressionWorkflow


def do_beta_regression(app_metadata: cli_tools.Metadata,
                       regression_specification: RegressionSpecification):
    logger.debug('Starting beta regression.')
    # init high level objects
    data_interface = RegressionDataInterface.from_specification(regression_specification)

    # build directory structure
    location_ids = data_interface.load_location_ids()
    data_interface.regression_paths.make_dirs(location_ids)
    # Fixme: Inconsistent data writing interfaces
    regression_specification.dump(data_interface.regression_paths.regression_specification)

    # Check to make sure we have all the covariates we need
    data_interface.check_covariates(regression_specification.covariates)
    # build workflow and launch
    regression_wf = RegressionWorkflow(regression_specification.data.output_root)
    n_draws = data_interface.get_draw_count()
    regression_wf.attach_beta_regression_tasks(n_draws=n_draws)
    regression_wf.run()
