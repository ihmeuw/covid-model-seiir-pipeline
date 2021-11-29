from typing import Dict

import pandas as pd

from covid_model_seiir_pipeline.lib import (
    io,
)
from covid_model_seiir_pipeline.pipeline.preprocessing import (
    PreprocessingSpecification,
    PreprocessingDataInterface,
)
from covid_model_seiir_pipeline.pipeline.fit.specification import (
    FitSpecification,
)


class FitDataInterface:

    def __init__(self,
                 preprocessing_root: io.PreprocessingRoot,
                 fit_root: io.FitRoot):
        self.preprocessing_root = preprocessing_root
        self.fit_root = fit_root

        self._preprocessing_data_interface = None

    @property
    def preprocessing_data_interface(self) -> PreprocessingDataInterface:
        if self._preprocessing_data_interface is None:
            specification = PreprocessingSpecification.from_dict(io.load(self.preprocessing_root.specification()))
            self._preprocessing_data_interface = PreprocessingDataInterface.from_specification(specification)
        return self._preprocessing_data_interface

    @classmethod
    def from_specification(cls, specification: FitSpecification) -> 'FitDataInterface':
        preprocessing_spec = PreprocessingSpecification.from_version_root(specification.data.seir_preprocess_version)
        preprocessing_root = io.PreprocessingRoot(preprocessing_spec.data.output_root,
                                                  data_format=preprocessing_spec.data.output_format)
        return cls(
            preprocessing_root=preprocessing_root,
            fit_root=io.FitRoot(specification.data.output_root,
                                data_format=specification.data.output_format),
        )

    def make_dirs(self, **prefix_args) -> None:
        io.touch(self.fit_root, **prefix_args)

    def get_n_draws(self):
        specification = self.load_specification()
        fit_draws = specification.data.n_draws
        preprocess_draws = self.preprocessing_data_interface.get_n_draws()
        if not fit_draws <= preprocess_draws:
            raise ValueError(f"Can't run fit with more draws than preprocessing.\n"
                             f"Fit draws requested: {fit_draws}. Preprocessing draws: {preprocess_draws}.")
        return fit_draws

    #####################
    # Preprocessed Data #
    #####################

    def load_hierarchy(self, name: str) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_hierarchy(name=name)

    def load_population(self, measure: str) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_population(measure=measure)

    def load_age_patterns(self) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_age_patterns()

    def load_reported_epi_data(self) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_reported_epi_data()

    def load_total_covid_scalars(self, draw_id: int = None) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_total_covid_scalars(draw_id=draw_id)

    def load_seroprevalence(self, draw_id: int = None) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_seroprevalence(draw_id=draw_id)

    def load_sensitivity(self, draw_id: int = None) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_sensitivity(draw_id)

    def load_testing_data(self) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_testing_data()

    def load_covariate(self, covariate: str, scenario: str) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_covariate(covariate, scenario)

    def load_covariate_info(self, covariate: str, info_type: str) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_covariate_info(covariate, info_type)

    def load_variant_prevalence(self, scenario: str) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_variant_prevalence(scenario)

    def load_waning_parameters(self, measure: str) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_waning_parameters(measure)

    def load_vaccine_uptake(self, scenario: str) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_vaccine_uptake(scenario)

    def load_vaccine_risk_reduction(self, scenario: str) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_vaccine_risk_reduction(scenario)

    ################
    # Fit data I/O #
    ################

    def save_specification(self, specification: FitSpecification) -> None:
        io.dump(specification.to_dict(), self.fit_root.specification())

    def load_specification(self) -> FitSpecification:
        spec_dict = io.load(self.fit_root.specification())
        return FitSpecification.from_dict(spec_dict)

    def save_covariate_options(self, covariate_options: Dict) -> None:
        io.dump(covariate_options, self.fit_root.covariate_options())

    def load_covariate_options(self) -> Dict:
        return io.load(self.fit_root.covariate_options())

    def save_epi_measures(self, data: pd.DataFrame, draw_id: int):
        io.dump(data, self.fit_root.epi_measures(draw_id=draw_id))

    def load_epi_measures(self, draw_id: int):
        return io.load(self.fit_root.epi_measures(draw_id=draw_id))
