from typing import Dict, List

import numpy as np
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
                 preprocessing_data_interface: PreprocessingDataInterface,
                 fit_root: io.FitRoot):
        self.preprocessing_data_interface = preprocessing_data_interface
        self.fit_root = fit_root

    @classmethod
    def from_specification(cls, specification: FitSpecification) -> 'FitDataInterface':
        preprocessing_spec = PreprocessingSpecification.from_version_root(specification.data.seir_preprocess_version)
        preprocessing_data_interface = PreprocessingDataInterface.from_specification(preprocessing_spec)
        return cls(
            preprocessing_data_interface=preprocessing_data_interface,
            fit_root=io.FitRoot(specification.data.output_root,
                                data_format=specification.data.output_format),
        )

    def make_dirs(self, **prefix_args) -> None:
        io.touch(self.fit_root, **prefix_args)

    def get_n_draws(self):
        return self.preprocessing_data_interface.get_n_draws()

    def get_n_oversample_draws(self):
        return self.preprocessing_data_interface.get_n_oversample_draws()

    def get_n_total_draws(self):
        return self.preprocessing_data_interface.get_n_total_draws()

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

    def load_hospital_census_data(self) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_hospital_census_data()

    def load_hospital_bed_capacity(self) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_hospital_bed_capacity()

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

    def load_testing_capacity(self) -> pd.Series:
        testing_data = self.preprocessing_data_interface.load_testing_data()
        testing_capacity = testing_data['testing_capacity']
        testing_capacity_offset = 1
        testing_capacity += testing_capacity_offset
        return testing_capacity

    def load_covariate(self, covariate: str, scenario: str) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_covariate(covariate, scenario)

    def load_covariate_info(self, covariate: str, info_type: str) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_covariate_info(covariate, info_type)

    def load_variant_prevalence(self, scenario: str) -> pd.DataFrame:
        first_omega_date = self.load_specification().fit_parameters.omega_invasion_date
        data = self.preprocessing_data_interface.load_variant_prevalence(scenario)
        if not first_omega_date:
            return data

        first_omicron_date = data[data.omicron > 0.01].reset_index().date.min()
        first_omega_date = pd.Timestamp(first_omega_date)
        shift = (first_omega_date - first_omicron_date).days
        final_outputs = []
        for location_id in data.reset_index().location_id.unique():
            omicron = data.loc[location_id, 'omicron']
            omega = omicron.shift(shift).ffill().bfill()
            new_data = data.loc[location_id].drop(columns=['omega', 'omicron'])
            new_data['omicron'] = np.minimum(1 - omega, omicron)
            new_data['omega'] = omega
            new_data = new_data.div(new_data.sum(axis=1), axis=0)
            new_data['location_id'] = location_id
            final_outputs.append(new_data.reset_index().set_index(['location_id', 'date']))
        final_data = pd.concat(final_outputs).sort_index()
        return final_data

    def load_waning_parameters(self, measure: str) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_waning_parameters(measure)

    def load_vaccine_summary(self, columns: List[str] = None) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_vaccine_summary(columns=columns)

    def load_vaccine_uptake(self, scenario: str) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_vaccine_uptake(scenario)

    def load_vaccine_risk_reduction(self, scenario: str) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_vaccine_risk_reduction(scenario)

    def load_antiviral_coverage(self, scenario: str) -> pd.DataFrame:
        return self.preprocessing_data_interface.load_antiviral_coverage(scenario)

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
        covariate_options = io.load(self.fit_root.covariate_options())
        if draw_id is not None:
            covariate_options = {rate: draws[draw_id]
                                 for rate, draws in covariate_options.items()}
        return covariate_options

    def load_covariate_pool(self, draw_id: int, measure: str) -> pd.DataFrame:
        # Be careful about imports. `data` really shouldn't depend on `model` but
        # it's also the right place for this constant to live for now.
        from covid_model_seiir_pipeline.pipeline.fit.model.covariate_pool import COVARIATE_POOL
        rate = {
            'death': 'ifr',
            'admission': 'ihr',
            'case': 'idr'
        }[measure]
        covariate_options = self.load_covariate_options(draw_id)[rate]
        covariate_pool = []
        for covariate in COVARIATE_POOL:
            cov = (self.load_covariate(covariate, 'reference')
                   .groupby('location_id')[f'{covariate}_reference']
                   .mean()
                   .rename(covariate))
            covariate_pool.append(cov)
        covariate_pool = pd.concat(covariate_pool, axis=1).loc[:, covariate_options]
        return covariate_pool

    def save_draw_resampling_map(self, draw_resampling: Dict) -> None:
        io.dump(draw_resampling, self.fit_root.draw_resampling())

    def load_draw_resampling_map(self) -> Dict:
        return io.load(self.fit_root.draw_resampling())

    def save_fit_failures(self, fit_failures: pd.DataFrame) -> None:
        io.dump(fit_failures, self.fit_root.fit_failures())

    def load_fit_failures(self) -> pd.DataFrame:
        return io.load(self.fit_root.fit_failures())

    def save_fit_residuals(self, fit_residuals: pd.DataFrame) -> None:
        io.dump(fit_residuals, self.fit_root.fit_residuals())

    def load_fit_residuals(self) -> pd.DataFrame:
        return io.load(self.fit_root.fit_residuals())

    def save_ode_params(self, data: pd.Series, draw_id: int, measure_version: str) -> None:
        io.dump(data, self.fit_root.ode_parameters(measure=measure_version, draw_id=draw_id))

    def load_ode_params(self, draw_id: int, measure_version: str = 'final', columns: List[str] = None) -> pd.DataFrame:
        return io.load(self.fit_root.ode_parameters(measure=measure_version, draw_id=draw_id, columns=columns))

    def save_phis(self, data: pd.Series, draw_id: int) -> None:
        io.dump(data, self.fit_root.phis(draw_id=draw_id))

    def load_phis(self, draw_id: int) -> pd.DataFrame:
        return io.load(self.fit_root.phis(draw_id=draw_id))

    def save_input_epi_measures(self, data: pd.DataFrame, draw_id: int, measure_version: str):
        io.dump(data, self.fit_root.input_epi_measures(measure=measure_version, draw_id=draw_id))

    def load_input_epi_measures(self, draw_id: int, measure_version: str = 'final', columns: List[str] = None) -> pd.DataFrame:
        return io.load(self.fit_root.input_epi_measures(measure=measure_version, draw_id=draw_id, columns=columns))

    def save_rates_data(self, data: pd.DataFrame, draw_id: int, measure_version: str) -> None:
        io.dump(data, self.fit_root.rates_data(measure=measure_version, draw_id=draw_id))

    def load_rates_data(self, draw_id: int, measure_version: str, columns: List[str] = None) -> pd.DataFrame:
        return io.load(self.fit_root.rates_data(measure=measure_version, draw_id=draw_id, columns=columns))

    def save_rates(self, data: pd.DataFrame, draw_id: int, measure_version: str) -> None:
        io.dump(data, self.fit_root.rates(measure=measure_version, draw_id=draw_id))

    def load_rates(self, draw_id: int, measure_version: str = 'final', columns: List[str] = None) -> pd.DataFrame:
        return io.load(self.fit_root.rates(measure=measure_version, draw_id=draw_id, columns=columns))

    def save_posterior_epi_measures(self, data: pd.DataFrame, draw_id: int, measure_version: str):
        io.dump(data, self.fit_root.posterior_epi_measures(measure=measure_version, draw_id=draw_id))

    def load_posterior_epi_measures(self, draw_id: int, measure_version: str = 'final', columns: List[str] = None) -> pd.DataFrame:
        return io.load(self.fit_root.posterior_epi_measures(measure=measure_version, draw_id=draw_id, columns=columns))

    def save_compartments(self, data: pd.DataFrame, draw_id: int, measure_version: str) -> None:
        io.dump(data, self.fit_root.compartments(measure=measure_version, draw_id=draw_id))

    def load_compartments(self, draw_id: int, measure_version: str = 'final', columns: List[str] = None) -> pd.DataFrame:
        return io.load(self.fit_root.compartments(measure=measure_version, draw_id=draw_id, columns=columns))

    def save_fit_beta(self, data: pd.DataFrame, draw_id: int, measure_version: str) -> None:
        io.dump(data, self.fit_root.beta(measure=measure_version, draw_id=draw_id))

    def load_fit_beta(self, draw_id: int, measure_version: str = 'final', columns: List[str] = None) -> pd.DataFrame:
        return io.load(self.fit_root.beta(measure=measure_version, draw_id=draw_id, columns=columns))

    def save_final_seroprevalence(self, data: pd.DataFrame, measure_version: str, draw_id: int) -> None:
        io.dump(data, self.fit_root.seroprevalence(measure=measure_version, draw_id=draw_id))

    def load_final_seroprevalence(self, draw_id: int, measure_version: str, columns: List[str] = None) -> pd.DataFrame:
        return io.load(self.fit_root.seroprevalence(measure=measure_version, draw_id=draw_id, columns=columns))

    def save_summary(self, data: pd.DataFrame, measure: str) -> None:
        io.dump(data, self.fit_root.summary(measure=measure))

    def load_summary(self, measure: str) -> pd.DataFrame:
        return io.load(self.fit_root.summary(measure=measure))
