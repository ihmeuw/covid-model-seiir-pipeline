from __future__ import annotations

from typing import Union
from pathlib import Path

from seiir_model_pipeline.exceptions import VersionAlreadyExists
from seiir_model_pipeline.utilities import VersionDirectory
from seiir_model_pipeline.regression.covariate import CovariateFormatter
from seiir_model_pipeline.regression.data import RegressionDirectories
from seiir_model_pipeline.regression.specification import (RegressionSpecification,
                                                           dump_regression_specification,
                                                           load_regression_specification)


class RegressionVersion:

    @classmethod
    def init_version(cls, regression_specification: RegressionSpecification,
                     regression_dir: Path
                     ) -> RegressionVersion:
        """initialization logic goes here"""
        spec_path = regression_dir / "regression_specification.yaml"

        if spec_path.exists():
            ver_dir = VersionDirectory(version_dir=regression_dir)
            raise VersionAlreadyExists(f"A version ({ver_dir.version_name}) already exists in "
                                       f"directory ({ver_dir.root_dir}). Either remove it or "
                                       f"run using resume.")

        # make input directories
        regression_directories = RegressionDirectories(
            regression_dir=regression_dir,
            infection_dir=Path(regression_specification.data.infection_version)
        )
        regression_directories.make_dirs()

        # etl covariate data
        cov_etl = CovariateFormatter(
            covariate_dir=Path(regression_specification.data.covariate_version),
            covariate_specifications=regression_specification.covariates,
            location_ids=[]  # TODO: locations in regression_spec?
        )
        cov_etl.etl_covariates(regression_specification.parameters, regression_directories)

        dump_regression_specification(spec_path)

        return cls(spec_path)

    def __init__(self, specification_path: Union[str, Path]) -> None:
        """resume logic goes here"""
        self.specification_path = Path(specification_path)
        self.specification = load_regression_specification(specification_path)
        self.directories = RegressionDirectories(
            regression_dir=Path(self.specification.data.output_root),
            infection_dir=Path(self.specification.data.infection_version)
        )
        # self.data_interface = RegressionDataInterface(self.directories)

    def run(self):
        pass
