from shutil import copyfile

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

    # copy info files
    for covariate in regression_specification.covariates.keys():
        info_files = data_interface.covariate_paths.get_info_files(covariate)
        for file in info_files:
            dest_path = data_interface.regression_paths.info_dir / file.name
            copyfile(str(file), str(dest_path))

    # build workflow and launch
    regression_wf = RegressionWorkflow(regression_specification.data.output_root)
    n_draws = data_interface.get_draw_count()
    regression_wf.attach_beta_regression_tasks(n_draws=n_draws)
    regression_wf.run()
