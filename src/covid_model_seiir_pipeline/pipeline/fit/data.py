from typing import Dict, List

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
        specification = self.load_specification()
        data = self.preprocessing_data_interface.load_total_covid_scalars(draw_id=draw_id)
        if specification.rates_parameters.mortality_scalar == 'total':
            return data
        elif specification.rates_parameters.mortality_scalar == 'unscaled':
            data.loc[:, :] = 1.0
            return data
        else:
            raise ValueError(f'Unknown scaling option {specification.rates_parameters.mortality_scalar}')

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

    def load_covariate_options(self, draw_id: int = None) -> Dict:
        covariate_pool = io.load(self.fit_root.covariate_options())
        if draw_id is not None:
            covariate_pool = {rate: draws[draw_id] for rate, draws in covariate_pool.items()}
        return covariate_pool

    def save_ode_params(self, data: pd.Series, draw_id: int) -> None:
        io.dump(data, self.fit_root.ode_parameters(draw_id=draw_id))

    def load_ode_params(self, draw_id: int) -> None:
        return io.load(self.fit_root.ode_parameters(draw_id=draw_id))

    def save_input_epi_measures(self, data: pd.DataFrame, draw_id: int):
        io.dump(data, self.fit_root.input_epi_measures(draw_id=draw_id))

    def load_input_epi_measures(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return io.load(self.fit_root.input_epi_measures(draw_id=draw_id, columns=columns))

    def save_rates(self, data: pd.DataFrame, draw_id: int) -> None:
        io.dump(data, self.fit_root.rates(draw_id=draw_id))

    def load_rates(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return io.load(self.fit_root.rates(draw_id=draw_id, columns=columns))

    def save_posterior_epi_measures(self, data: pd.DataFrame, draw_id: int):
        io.dump(data, self.fit_root.posterior_epi_measures(draw_id=draw_id))

    def load_posterior_epi_measures(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return io.load(self.fit_root.posterior_epi_measures(draw_id=draw_id, columns=columns))

    def save_compartments(self, data: pd.DataFrame, draw_id: int) -> None:
        io.dump(data, self.fit_root.compartments(draw_id=draw_id))

    def load_compartments(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return io.load(self.fit_root.compartments(draw_id=draw_id, columns=columns))

    def save_beta(self, data: pd.DataFrame, draw_id: int) -> None:
        io.dump(data, self.fit_root.beta(draw_id=draw_id))

    def load_beta(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return io.load(self.fit_root.beta(draw_id=draw_id, columns=columns))

    def save_final_seroprevalence(self, data: pd.DataFrame, draw_id: int) -> None:
        io.dump(data, self.fit_root.seroprevalence(draw_id=draw_id))

    def load_final_seroprevalence(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return io.load(self.fit_root.seroprevalence(draw_id=draw_id, columns=columns))


