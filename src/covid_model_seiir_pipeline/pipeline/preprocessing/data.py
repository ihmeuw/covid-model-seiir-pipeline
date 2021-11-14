from pathlib import Path
from typing import Optional, Union

import pandas as pd

from covid_model_seiir_pipeline.lib import (
    io,
    utilities,
)

from covid_model_seiir_pipeline.pipeline.preprocessing.specification import (
    PreprocessingSpecification,
)


class PreprocessingDataInterface:

    def __init__(self,
                 model_inputs_root: io.ModelInputsRoot,
                 age_specific_rates_root: io.AgeSpecificRatesRoot,
                 mortality_scalars_root: io.MortalityScalarsRoot,
                 mask_use_root: io.MaskUseRoot,
                 preprocessing_root: io.PreprocessingRoot):
        self.model_inputs_root = model_inputs_root
        self.age_specific_rates_root = age_specific_rates_root
        self.mortality_scalars_root = mortality_scalars_root
        self.mask_use_root = mask_use_root
        self.preprocessing_root = preprocessing_root

    @classmethod
    def from_specification(cls, specification: PreprocessingSpecification) -> 'PreprocessingDataInterface':
        return cls(
            model_inputs_root=io.ModelInputsRoot(specification.data.model_inputs_version),
            age_specific_rates_root=io.AgeSpecificRatesRoot(specification.data.age_specific_rates_version),
            mortality_scalars_root=io.MortalityScalarsRoot(specification.data.mortality_scalars_version),
            mask_use_root=io.MaskUseRoot(specification.data.mask_use_version),
            preprocessing_root=io.PreprocessingRoot(specification.data.output_root,
                                                    data_format=specification.data.output_format),
        )

    def make_dirs(self, **prefix_args) -> None:
        io.touch(self.preprocessing_root, **prefix_args)

    ####################
    # Metadata loaders #
    ####################

    def get_n_draws(self) -> int:
        specification = self.load_specification()
        return specification.data.n_draws

    #########################
    # Raw location handling #
    #########################

    @staticmethod
    def load_hierarchy_from_primary_source(location_set_version_id: Optional[int],
                                           location_file: Optional[Union[str, Path]]) -> pd.DataFrame:
        """Retrieve a location hierarchy from a file or from GBD if specified."""
        location_metadata = utilities.load_location_hierarchy(
            location_set_version_id=location_set_version_id,
            location_file=location_file,
        )
        return location_metadata

    ########################
    # Model inputs loaders #
    ########################

    ##############################
    # Age-specific Rates loaders #
    ##############################

    def load_age_pattern_data(self) -> pd.DataFrame:
        measure_map = {
            'ihr': ('hir_preds_5yr', 'hir'),
            'ifr': ('ifr_preds_5yr_global', 'ifr'),
            'seroprevalence': ('seroprev_preds_5yr', 'seroprev'),
        }
        measure_data = []
        for measure, (file_name, column_name) in measure_map:
            column_map = {
                'age_group_start': 'age_group_years_start',
                'age_group_end': 'age_group_years_end',
                column_name: measure,
            }
            data = io.load(self.age_specific_rates_root.rates_data(measure=file_name))
            data = data.rename(columns=column_map).loc[:, column_map.values()]
            data['age_group_years_end'].iloc[-1] = 125
            measure_data.append(data)
        measure_data = pd.concat(measure_data, axis=1)
        measure_data['key'] = 1

        modeling_hierarchy = self.load_modeling_hierarchy().reset_index()
        modeling_hierarchy['key'] = 1

        # Broadcast over location id.
        measure_data = (modeling_hierarchy
                        .loc[:, ['location_id', 'key']]
                        .merge(measure_data)
                        .set_index(['location_id', 'age_group_years_start', 'age_group_years_end'])
                        .sort_index()
                        .drop(columns='key'))

        return measure_data

    #########################
    # Mortality Scalar Data #
    #########################

    def load_raw_total_covid_scalars(self) -> pd.DataFrame:
        data = io.load(self.mortality_scalars_root.total_covid_draw())
        data['draw'] -= 1
        data = data.set_index('draw', append=True).sort_index().unstack()
        data.columns = [f'draw_{d}' for d in data.columns.droplevel()]
        return data

    #################
    # Mask Use Data #
    #################

    def load_raw_mask_use_data(self, scenario: str):
        try:
            scenario_file = {
                'reference': 'mask_use',
                'best': 'mask_use_best',
                'worse': 'mask_use_worse',
            }[scenario]
        except KeyError:
            raise ValueError(f'Unknown mask use scenario {scenario}.')
        key = 'mask_best' if scenario == 'best' else 'mask_use'
        data = io.load(self.mask_use_root.mask_use(measure=scenario_file))
        data = data.loc[:, ['observed', key]].rename(columns={key: f'mask_use_{scenario}'})
        return data

    ##########################
    # Preprocessing data I/O #
    ##########################

    def save_specification(self, specification: PreprocessingSpecification) -> None:
        io.dump(specification.to_dict(), self.preprocessing_root.specification())

    def load_specification(self) -> PreprocessingSpecification:
        spec_dict = io.load(self.preprocessing_root.specification())
        return PreprocessingSpecification.from_dict(spec_dict)

    def save_modeling_hierarchy(self, hierarchy: pd.DataFrame) -> None:
        io.dump(hierarchy, self.preprocessing_root.hierarchy())

    def load_modeling_hierarchy(self) -> pd.DataFrame:
        return io.load(self.preprocessing_root.hierarchy())

    def save_age_patterns(self, age_patterns: pd.DataFrame) -> None:
        io.dump(age_patterns, self.preprocessing_root.age_patterns())

    def load_age_patterns(self) -> pd.DataFrame:
        return io.load(self.preprocessing_root.age_patterns())

    def save_total_covid_scalars(self, data: pd.DataFrame):
        io.dump(data, self.preprocessing_root.total_covid_scalars())

    def load_total_covid_scalars(self, draw_id: int = None) -> pd.DataFrame:
        columns = [f'draw_{draw_id}'] if draw_id is not None else None
        return io.load(self.preprocessing_root.total_covid_scalars(columns=columns))

    def save_mask_use(self, data: pd.DataFrame, scenario: str):
        io.dump(data, self.preprocessing_root.mask_use(scenario=scenario))
