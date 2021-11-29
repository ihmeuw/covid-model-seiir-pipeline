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

    def load_modeling_hierarchy(self) -> pd.DataFrame:
        return io.load(self.preprocessing_root.hierarchy())

    def load_population(self, measure: str) -> pd.DataFrame:
        return io.load(self.preprocessing_root.population(measure=measure))

    def load_age_patterns(self) -> pd.DataFrame:
        return io.load(self.preprocessing_root.age_patterns())

    def load_total_covid_scalars(self, draw_id: int = None) -> pd.DataFrame:
        columns = [f'draw_{draw_id}'] if draw_id is not None else None
        return io.load(self.preprocessing_root.total_covid_scalars(columns=columns))

    def load_global_serology(self) -> pd.DataFrame:
        return io.load(self.preprocessing_root.global_serology())

    def load_testing_data(self) -> pd.DataFrame:
        return io.load(self.preprocessing_root.testing_for_idr())

    def load_covariate(self, covariate: str, scenario: str) -> pd.DataFrame:
        return io.load(self.preprocessing_root[covariate](covariate_scenario=scenario))

    def load_covariate_info(self, covariate: str, info_type: str) -> pd.DataFrame:
        return io.load(self.preprocessing_root[f"{covariate}_info"](info_type=info_type))

    def load_variant_prevalence(self, scenario: str) -> pd.DataFrame:
        return io.load(self.preprocessing_root.variant_prevalence(scenario=scenario))

    def load_waning_parameters(self, measure: str) -> pd.DataFrame:
        return io.load(self.preprocessing_root.waning_parameters(measure=measure))

    def load_vaccine_uptake(self, scenario: str) -> pd.DataFrame:
        return io.load(self.preprocessing_root.vaccine_uptake(covariate_scenario=scenario))

    def load_vaccine_risk_reduction(self, scenario: str) -> pd.DataFrame:
        return io.load(self.preprocessing_root.vaccine_risk_reduction(covariate_scenario=scenario))

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
