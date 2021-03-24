from functools import reduce
from pathlib import Path
from typing import List, Iterable, Optional, Union

from loguru import logger
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    io,
    utilities,
)
from covid_model_seiir_pipeline.pipeline.fit_oos.specification import (
    FitSpecification,
)


class FitDataInterface:

    def __init__(self,
                 infection_root: io.InfectionRoot,
                 covariate_root: io.CovariateRoot,
                 variant_root: io.VariantRoot,
                 fit_root: io.FitRoot):
        self.infection_root = infection_root
        self.covariate_root = covariate_root
        self.variant_root = variant_root
        self.fit_root = fit_root

    @classmethod
    def from_specification(cls, specification: FitSpecification) -> 'FitDataInterface':
        infection_root = io.InfectionRoot(specification.data.infection_version)
        covariate_root = io.CovariateRoot(specification.data.covariate_version)
        variant_root = io.VariantRoot(specification.data.variant_version)
        fit_root = io.FitRoot(specification.data.output_root,
                              data_format=specification.data.output_format)

        return cls(
            infection_root=infection_root,
            covariate_root=covariate_root,
            variant_root=variant_root,
            fit_root=fit_root,
        )

    def make_dirs(self, **prefix_args) -> None:
        io.touch(self.fit_root, **prefix_args)

    def get_n_draws(self) -> int:
        fit_spec = io.load(self.fit_root.specification())
        return fit_spec['data']['n_draws']

    #########################
    # Raw location handling #
    #########################

    @staticmethod
    def load_hierarchy_from_primary_source(location_set_version_id: Optional[int],
                                           location_file: Optional[Union[str, Path]]) -> Union[pd.DataFrame, None]:
        """Retrieve a location hierarchy from a file or from GBD if specified."""
        if location_set_version_id:
            location_metadata = utilities.load_location_hierarchy(location_set_id=111,
                                                                  location_set_version_id=location_set_version_id)
        elif location_file:
            location_metadata = utilities.load_location_hierarchy(location_file=location_file)
        else:
            location_metadata = None

        return location_metadata

    def filter_location_ids(self, desired_location_hierarchy: pd.DataFrame = None) -> List[int]:
        draw_0_data = self.load_full_past_infection_data(draw_id=0)
        total_deaths = draw_0_data.groupby('location_id').deaths.sum()
        modeled_locations = total_deaths[total_deaths > 5].index.tolist()

        variant_locs = self.get_escape_variant_special_locs()
        modeled_locations = list(set(modeled_locations).intersection(variant_locs))

        if desired_location_hierarchy is None:
            desired_locations = modeled_locations
        else:
            most_detailed = desired_location_hierarchy.most_detailed == 1
            desired_locations = desired_location_hierarchy.loc[most_detailed, 'location_id'].tolist()

        missing_locations = list(set(desired_locations).difference(modeled_locations))
        if missing_locations:
            logger.warning("Some locations present in location metadata are missing from the "
                           f"infection models. Missing locations are {sorted(missing_locations)}.")
        return list(set(desired_locations).intersection(modeled_locations))

    ##########################
    # Infection data loaders #
    ##########################

    def load_full_past_infection_data(self, draw_id: int) -> pd.DataFrame:
        infection_data = io.load(self.infection_root.infections(draw_id=draw_id))
        infection_data = (infection_data
                          .loc[:, ['infections_draw', 'deaths']]
                          .rename(columns={'infections_draw': 'infections'}))
        return infection_data

    def load_past_infection_data(self, draw_id: int):
        location_ids = self.load_location_ids()
        infection_data = self.load_full_past_infection_data(draw_id=draw_id)
        return infection_data.loc[location_ids]

    def load_covariate(self, covariate: str) -> pd.DataFrame:
        location_ids = self.load_location_ids()
        covariate_df = io.load(self.covariate_root[covariate](covariate_scenario='reference'))
        covariate_df = covariate_df.loc[location_ids].rename(columns={f'{covariate}_reference': covariate})
        return covariate_df.loc[:, [covariate]]

    def load_vaccine_info(self, vaccine_scenario: str):
        location_ids = self.load_location_ids()
        info_df = io.load(self.covariate_root.vaccine_info(info_type=f'vaccinations_{vaccine_scenario}'))
        return self._format_covariate_data(info_df, location_ids)

    ##############################
    # Miscellaneous data loaders #
    ##############################

    def get_infections_metadata(self):
        return io.load(self.infection_root.metadata())

    def get_model_inputs_metadata(self):
        infection_metadata = self.get_infections_metadata()
        return infection_metadata['model_inputs_metadata']

    def load_population(self) -> pd.DataFrame:
        metadata = self.get_model_inputs_metadata()
        model_inputs_version = metadata['output_path']
        population_path = Path(model_inputs_version) / 'output_measures' / 'population' / 'all_populations.csv'
        population_data = pd.read_csv(population_path)
        return population_data

    def load_five_year_population(self) -> pd.DataFrame:
        population = self.load_population()
        location_ids = self.load_location_ids()
        in_locations = population['location_id'].isin(location_ids)
        is_2019 = population['year_id'] == 2019
        is_both_sexes = population['sex_id'] == 3
        five_year_bins = [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 235]
        is_five_year_bins = population['age_group_id'].isin(five_year_bins)
        population = population.loc[in_locations & is_2019 & is_both_sexes & is_five_year_bins, :]
        return population

    def load_total_population(self) -> pd.Series:
        population = self.load_five_year_population()
        population = population.groupby('location_id')['population'].sum()
        return population

    def get_escape_variant_special_locs(self):
        b1351 = io.load(self.variant_root.original_data(measure='Special_B1351_Final_2021_03_21.csv'))
        b1351_locs = b1351.loc_id.unique().tolist()
        p1 = io.load(self.variant_root.original_data(measure='Special_P1_Final_2021_03_21.csv'))
        p1_locs = p1.loc_id.unique().tolist()
        return list(set(b1351_locs + p1_locs))

    def load_variant_prevalence(self):
        b117 = self.load_covariate('variant_prevalence_B117').variant_prevalence_B117
        b1351 = self.load_covariate('variant_prevalence_B1351').variant_prevalence_B1351
        p1 = self.load_covariate('variant_prevalence_P1').variant_prevalence_P1
        rho_variant = (b1351 + p1).rename('rho_variant')
        rho = (b117 / (1 - rho_variant)).fillna(0).rename('rho')
        return pd.concat([rho, rho_variant], axis=1)

    #######################
    # Fit data I/O #
    #######################

    def save_specification(self, specification: FitSpecification) -> None:
        io.dump(specification.to_dict(), self.fit_root.specification())

    def load_specification(self) -> FitSpecification:
        spec_dict = io.load(self.fit_root.specification())
        return FitSpecification.from_dict(spec_dict)

    def save_location_ids(self, location_ids: List[int]) -> None:
        io.dump(location_ids, self.fit_root.locations())

    def load_location_ids(self) -> List[int]:
        return io.load(self.fit_root.locations())

    def save_hierarchy(self, hierarchy: pd.DataFrame) -> None:
        io.dump(hierarchy, self.fit_root.hierarchy())

    def load_hierarchy(self) -> pd.DataFrame:
        return io.load(self.fit_root.hierarchy())

    def save_betas(self, betas: pd.DataFrame, draw_id: int) -> None:
        io.dump(betas, self.fit_root.beta(draw_id=draw_id))

    def load_betas(self, draw_id: int) -> pd.DataFrame:
        return io.load(self.fit_root.beta(draw_id=draw_id))

    def save_compartments(self, compartments: pd.DataFrame, draw_id: int) -> None:
        io.dump(compartments, self.fit_root.compartments(draw_id=draw_id))

    def load_compartments(self, draw_id: int) -> pd.DataFrame:
        return io.load(self.fit_root.compartments(draw_id=draw_id))

    def save_ode_parameters(self, df: pd.DataFrame, draw_id: int) -> None:
        io.dump(df, self.fit_root.ode_parameters(draw_id=draw_id))

    def load_ode_parameters(self, draw_id: int) -> pd.DataFrame:
        return io.load(self.fit_root.ode_parameters(draw_id=draw_id))

    #########################
    # Non-interface helpers #
    #########################

    @staticmethod
    def _format_covariate_data(dataset: pd.DataFrame, location_ids: List[int], with_observed: bool = False):
        shared_locs = list(set(dataset.index.get_level_values('location_id')).intersection(location_ids))
        dataset = dataset.loc[shared_locs]
        if with_observed:
            dataset = dataset.set_index('observed', append=True)
        return dataset
