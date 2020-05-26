from pathlib import Path
from shutil import copyfile

from covid_shared import cli_tools

from seiir_model_pipeline.paths import RegressionPaths
from seiir_model_pipeline.regression.specification import RegressionSpecification
from seiir_model_pipeline.regression.data import RegressionDataInterface
from seiir_model_pipeline.regression.workflow import RegressionWorkflow


def do_beta_regression(app_metadata: cli_tools.Metadata,
                       regression_specification: RegressionSpecification,
                       output_dir: Path):
    # init high level objects
    regression_paths = RegressionPaths(output_dir, read_only=False)
    data_interface = RegressionDataInterface(regression_specification.data)

    # build directory structure
    location_ids = data_interface.load_location_ids()
    regression_paths.make_dirs(location_ids)

    # copy info files
    for covariate in regression_specification.covariates.keys():
        info_files = data_interface.covariate_paths.get_info_files(covariate)
        for file in info_files:
            dest_path = regression_paths.info_dir / file.name
            copyfile(str(file), str(dest_path))

    # build workflow and launch
    regression_wf = RegressionWorkflow(regression_specification)
    regression_wf.attach_regression_tasks()
    regression_wf.run()
