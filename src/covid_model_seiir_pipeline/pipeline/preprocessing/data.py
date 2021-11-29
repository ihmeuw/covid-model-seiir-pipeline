from pathlib import Path
from typing import Dict, Optional, Tuple, Union

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
                 mobility_root: io.MobilityRoot,
                 pneumonia_root: io.PneumoniaRoot,
                 population_density_root: io.PopulationDensityRoot,
                 testing_root: io.TestingRoot,
                 variant_prevalence_root: io.VariantPrevalenceRoot,
                 vaccine_coverage_root: io.VaccineCoverageRoot,
                 serology_vaccine_coverage_root: io.VaccineCoverageRoot,
                 vaccine_efficacy_root: io.VaccineEfficacyRoot,
                 preprocessing_root: io.PreprocessingRoot):
        self.model_inputs_root = model_inputs_root
        self.age_specific_rates_root = age_specific_rates_root
        self.mortality_scalars_root = mortality_scalars_root
        self.mask_use_root = mask_use_root
        self.mobility_root = mobility_root
        self.pneumonia_root = pneumonia_root
        self.population_density_root = population_density_root
        self.testing_root = testing_root
        self.variant_prevalence_root = variant_prevalence_root
        self.vaccine_coverage_root = vaccine_coverage_root
        self.serology_vaccine_coverage_root = serology_vaccine_coverage_root
        self.vaccine_efficacy_root = vaccine_efficacy_root
        self.preprocessing_root = preprocessing_root

    @classmethod
    def from_specification(cls, specification: PreprocessingSpecification) -> 'PreprocessingDataInterface':
        return cls(
            model_inputs_root=io.ModelInputsRoot(specification.data.model_inputs_version),
            age_specific_rates_root=io.AgeSpecificRatesRoot(specification.data.age_specific_rates_version),
            mortality_scalars_root=io.MortalityScalarsRoot(specification.data.mortality_scalars_version),
            mask_use_root=io.MaskUseRoot(specification.data.mask_use_outputs_version),
            mobility_root=io.MobilityRoot(specification.data.mobility_covariate_version),
            pneumonia_root=io.PneumoniaRoot(specification.data.pneumonia_version),
            population_density_root=io.PopulationDensityRoot(specification.data.population_density_version),
            testing_root=io.TestingRoot(specification.data.testing_outputs_version),
            variant_prevalence_root=io.VariantPrevalenceRoot(specification.data.variant_scaleup_version),
            vaccine_coverage_root=io.VaccineCoverageRoot(specification.data.vaccine_coverage_version),
            serology_vaccine_coverage_root=io.VaccineCoverageRoot(specification.data.serology_vaccine_coverage_version),
            vaccine_efficacy_root=io.VaccineEfficacyRoot(specification.data.vaccine_efficacy_version),
            preprocessing_root=io.PreprocessingRoot(specification.data.output_root,
                                                    data_format=specification.data.output_format),
        )

    def make_dirs(self, **prefix_args) -> None:
        # Yuck hack to make directories work
        from covid_model_seiir_pipeline.pipeline.preprocessing.model import COVARIATES
        # Need to add these to the data root, which happens dynamically on access.
        _ = [self.preprocessing_root[c] for c in COVARIATES]
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

    def load_raw_population(self, measure: str) -> pd.DataFrame:
        pop = io.load(self.model_inputs_root.population()).reset_index()
        is_2019 = pop['year_id'] == 2019
        is_both_sexes = pop['sex_id'] == 3
        five_year_bins = [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 235]
        is_five_year_bins = pop['age_group_id'].isin(five_year_bins)
        keep_cols = ['location_id', 'age_group_years_start', 'age_group_years_end', 'population']
        five_year = pop.loc[is_2019 & is_both_sexes & is_five_year_bins, keep_cols]

        if measure == 'all_population':
            data = pop
        elif measure == 'five_year':
            data = five_year
        elif measure == 'risk_group':
            pop_lr = five_year[five_year['age_group_years_start'] < 65].groupby('location_id')['population'].sum()
            pop_hr = five_year[five_year['age_group_years_start'] >= 65].groupby('location_id')['population'].sum()
            data = pd.concat([pop_lr.rename('lr'), pop_hr.rename('hr')], axis=1)
        elif measure == 'total':
            data = five_year.groupby('location_id')['population'].sum()
        else:
            raise ValueError(f'Unknown population measure {measure}.')

        return data

    def load_raw_serology_data(self):
        data = io.load(self.model_inputs_root.serology(measure='global_serology_summary'))
        return data.reset_index()

    def load_epi_measures(self) -> Dict[str, pd.Series]:
        full_data_extra_hospital = self._format_full_data(io.load(self.model_inputs_root.full_data_extra_hospital()))
        cases = full_data_extra_hospital.loc[:, 'cumulative_cases'].dropna()
        hospitalizations = full_data_extra_hospital.loc[:, 'cumulative_hospitalizations'].dropna()
        full_data = self._format_full_data(io.load(self.model_inputs_root.full_data()))
        deaths = full_data.loc[:, 'cumulative_deaths'].dropna()
        return {'cases': cases, 'hospitalizations': hospitalizations, 'deaths': deaths}

    def _format_full_data(self, data: pd.DataFrame) -> pd.DataFrame:
        col_map = {
            'Confirmed': 'cumulative_cases',
            'Hospitalizations': 'cumulative_hospitalizations',
            'Deaths': 'cumulative_deaths',
        }
        data = data.rename(columns=col_map)
        data['date'] = pd.to_datetime(data['Date'])
        data['location_id'] = data['location_id'].astype(int)
        data = (data
                .set_index(['location_id', 'date'])
                .sort_index()
                .loc[:, col_map.values()]
                .reset_index())
        return data

    def load_gbd_covariate(self, covariate: str, with_observed: bool = False) -> pd.DataFrame:
        data = io.load(self.model_inputs_root.gbd_covariate(measure=covariate))
        if not with_observed and 'observed' in data.columns:
            data = data.drop(columns='observed')

        data = data.rename(columns={covariate: f'{covariate}_reference'})
        return data

    def load_assay_map(self) -> pd.DataFrame:
        # FIXME: hack, different format.
        data_path = self.model_inputs_root._root / 'serology' / 'waning_immunity' / 'assay_map.xlsx'
        data = pd.read_excel(data_path)
        return data

    def load_sensitivity_data(self) -> Dict[str, pd.DataFrame]:
        # FIXME: hack, different format
        root = self.model_inputs_root._root / 'serology' / 'waning_immunity'
        path_map = {
            'peluso': ['peluso_assay_sensitivity'],
            'perez_saez': ['perez-saez_n-roche', 'perez-saez_rbd-euroimmun', 'perez-saez_rbd-roche'],
            'bond': ['bond'],
            'muecksch': ['muecksch'],
            'lumley': ['lumley_n-abbott', 'lumley_s-oxford']
        }
        return {k: pd.concat([pd.read_excel(root / f'{name}.xlsx') for name in paths]) for k, paths in path_map.items()}

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
        for measure, (file_name, column_name) in measure_map.items():
            column_map = {
                'age_group_start': 'age_group_years_start',
                'age_group_end': 'age_group_years_end',
                column_name: measure,
            }
            data = io.load(self.age_specific_rates_root.rates_data(measure=file_name))
            data = data.rename(columns=column_map).loc[:, column_map.values()]
            data['age_group_years_end'].iloc[-1] = 125
            data = data.set_index(['age_group_years_start', 'age_group_years_end'])
            measure_data.append(data)
        measure_data = pd.concat(measure_data, axis=1).reset_index()
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
        data = io.load(self.mask_use_root.mask_use_data(measure=scenario_file))
        data = data.loc[:, ['observed', key]].rename(columns={key: f'mask_use_{scenario}'})
        return data

    #################
    # Mobility Data #
    #################

    def load_raw_mobility(self, scenario: str) -> pd.DataFrame:
        data = io.load(self.mobility_root.mobility_data(measure=f'mobility_{scenario}')).reset_index()
        data['observed'] = (1 - data['type']).astype(int)
        data['location_id'] = data['location_id'].astype(int)
        output_columns = ['location_id', 'date', 'observed', f'mobility_{scenario}']
        data = (data.rename(columns={'mobility_forecast': f'mobility_{scenario}'})
                .loc[:, output_columns]
                .sort_values(['location_id', 'date']))
        return data

    def load_raw_percent_mandates(self, scenario: str) -> pd.DataFrame:
        metadata = io.load(self.mobility_root.metadata())
        path = Path(metadata['sd_lift_path']) / f'percent_mandates_{scenario}.csv'

        data = pd.read_csv(path)
        data['date'] = pd.to_datetime(data['date'])
        data['location_id'] = data['location_id'].astype(int)

        output_columns = ['location_id', 'date', 'percent', 'percent_mandates']
        data = data.loc[:, output_columns]

        return data.loc[:, output_columns]

    def load_raw_mobility_effect_sizes(self) -> pd.DataFrame:
        effect_cols = ['sd1', 'sd2', 'sd3', 'psd1', 'psd3', 'anticipate']
        output_columns = ['location_id'] + effect_cols

        data = io.load(self.mobility_root.mobility_data(measure='mobility_mandate_coefficients')).reset_index()
        data['location_id'] = data['location_id'].astype(int)
        data = (data.rename(columns={f'{effect}_eff': effect for effect in effect_cols})
                .loc[:, output_columns]
                .sort_values(['location_id']))
        return data

    ##################
    # Pneumonia data #
    ##################

    def load_raw_pneumonia_data(self) -> pd.DataFrame:
        data = io.load(self.pneumonia_root.pneumonia_data()).reset_index()
        data['observed'] = float('nan')
        data = (data
                .loc[:, ['date', 'location_id', 'observed', 'value']]
                .rename(columns={'value': 'pneumonia_reference'}))
        return data

    ###########################
    # Population density data #
    ###########################

    def load_raw_population_density_data(self) -> pd.DataFrame:
        data = io.load(self.population_density_root.population_density_data()).reset_index()
        data = data.set_index(['location_id', 'pop_density'])['pop_proportion'].unstack()
        return data

    ################
    # Testing data #
    ################

    def load_raw_testing_data(self) -> pd.DataFrame:
        data = io.load(self.testing_root.testing_data()).reset_index()
        data['observed'] = data['observed'].astype(int)
        data = data.set_index(['location_id', 'date', 'observed']).sort_index().reset_index()
        return data

    ###########################
    # Variant prevalence data #
    ###########################

    def load_raw_variant_prevalence(self, scenario: str) -> pd.DataFrame:
        data = io.load(self.variant_prevalence_root.prevalence(scenario=scenario)).reset_index()
        data = (data
                .set_index(['location_id', 'date', 'variant']).prevalence
                .sort_index()
                .unstack())
        return data

    #########################
    # Vaccine coverage data #
    #########################

    def load_raw_vaccine_uptake(self, scenario: str) -> pd.DataFrame:
        try:
            scenario_file = {
                'reference': 'last_shots_in_arm_by_brand_w_booster_reference_kid_low',
                'child_fast': 'last_shots_in_arm_by_brand_w_booster_reference_kid_high',
                'booster_high': 'last_shots_in_arm_by_brand_w_booster_optimal_kid_low',
                'optimistic': 'last_shots_in_arm_by_brand_w_booster_optimal_kid_high',
            }[scenario]
        except KeyError:
            raise ValueError(f'Unknown vaccine scenario {scenario}.')
        data = io.load(self.vaccine_coverage_root.brand_specific_coverage(measure=scenario_file))
        data = data.drop(columns='Unnamed: 0').reset_index()
        return data

    def load_serology_vaccine_coverage(self) -> pd.DataFrame:
        # FIXME: This goes away.
        data = io.load(self.serology_vaccine_coverage_root.old_vaccine_coverage())

        keep_columns = [
            # total vaccinated (all and by three groups)
            'cumulative_all_vaccinated',
            'cumulative_essential_vaccinated',
            'cumulative_adults_vaccinated',
            'cumulative_elderly_vaccinated',

            # total seroconverted (all and by three groups)
            'cumulative_all_effective',
            'cumulative_essential_effective',
            'cumulative_adults_effective',
            'cumulative_elderly_effective',

            # elderly (mutually exclusive)
            'cumulative_hr_effective_wildtype',
            'cumulative_hr_effective_protected_wildtype',
            'cumulative_hr_effective_variant',
            'cumulative_hr_effective_protected_variant',

            # other adults (mutually exclusive)
            'cumulative_lr_effective_wildtype',
            'cumulative_lr_effective_protected_wildtype',
            'cumulative_lr_effective_variant',
            'cumulative_lr_effective_protected_variant',
        ]

        return data.sort_index().loc[:, keep_columns]

    #########################
    # Vaccine efficacy data #
    #########################

    def load_efficacy_table(self) -> pd.DataFrame:
        data = io.load(self.vaccine_efficacy_root.efficacy_table())
        data = (data
                .rename(columns={'merge_name': 'brand'})
                .set_index('brand')
                .loc[:, ['efficacy', 'prop_protected_not_infectious', 'variant_efficacy', 'variant_infection']])
        # Use more sensible column names that care about variants and efficacy endpoints.
        data.columns = pd.MultiIndex.from_tuples(
            [('ancestral', 'severe_disease'),
             ('ancestral', 'infection'),
             ('delta', 'severe_disease'),
             ('delta', 'infection')],
            names=[None, 'endpoint']
        )
        # Pivot endpoint from the columns to the index.
        data = (data
                .stack()
                .reorder_levels(['endpoint', 'brand'])
                .sort_index())
        return data

    def load_waning_data(self) -> pd.DataFrame:
        data = io.load(self.vaccine_efficacy_root.waning_distribution(measure='vetsvacc'))
        pfizer = io.load(self.vaccine_efficacy_root.waning_distribution(measure='averageloglin'))
        data.loc[:, 'pfi_sev'] = pfizer['aver_pfi_sev']
        expected_cols = {
            'mid_point',
            'logpfi_inf', 'pfi_inf', 'logpfi_symp', 'pfi_symp', 'logpfi_sev', 'pfi_sev',
            'logmod_inf', 'mod_inf', 'logmod_symp', 'mod_symp', 'logmod_sev', 'mod_sev',
            'logast_symp', 'ast_symp', 'logast_sev', 'ast_sev',
        }
        assert set(data.columns) == expected_cols
        data = (data
                .rename(columns={'mid_point': 'weeks_since_delivery'})
                .set_index('weeks_since_delivery'))

        endpoint_map = {'inf': 'infection', 'sev': 'severe_disease', 'symp': 'symptomatic_infection', }

        out = []
        for col in data.columns:
            brand, endpoint = col.split('_')
            if brand[:3] == 'log':
                continue
            s = data[col].rename('value').to_frame()
            s['brand'] = brand
            s['endpoint'] = endpoint_map[endpoint]
            out.append(s)

        data = pd.concat(out).reset_index()
        data = data.set_index(['endpoint', 'brand', 'weeks_since_delivery']).sort_index().value
        return data

    ##########################
    # Preprocessing data I/O #
    ##########################

    def save_specification(self, specification: PreprocessingSpecification) -> None:
        io.dump(specification.to_dict(), self.preprocessing_root.specification())

    def load_specification(self) -> PreprocessingSpecification:
        spec_dict = io.load(self.preprocessing_root.specification())
        return PreprocessingSpecification.from_dict(spec_dict)

    def save_hierarchy(self, hierarchy: pd.DataFrame, name: str) -> None:
        io.dump(hierarchy, self.preprocessing_root.hierarchy(measure=name))

    def load_hierarchy(self, name: str) -> pd.DataFrame:
        return io.load(self.preprocessing_root.hierarchy(measure=name))

    def save_population(self, data: pd.DataFrame, measure: str) -> None:
        io.dump(data, self.preprocessing_root.population(measure=measure))

    def load_population(self, measure: str) -> pd.DataFrame:
        return io.load(self.preprocessing_root.population(measure=measure))

    def save_age_patterns(self, age_patterns: pd.DataFrame) -> None:
        io.dump(age_patterns, self.preprocessing_root.age_patterns())

    def load_age_patterns(self) -> pd.DataFrame:
        return io.load(self.preprocessing_root.age_patterns())

    def save_reported_epi_data(self, data: pd.DataFrame) -> None:
        io.dump(data, self.preprocessing_root.reported_epi_data())

    def load_reported_epi_data(self) -> pd.DataFrame:
        return io.load(self.preprocessing_root.reported_epi_data())

    def save_total_covid_scalars(self, data: pd.DataFrame):
        io.dump(data, self.preprocessing_root.total_covid_scalars())

    def load_total_covid_scalars(self, draw_id: int = None) -> pd.DataFrame:
        columns = [f'draw_{draw_id}'] if draw_id is not None else None
        return io.load(self.preprocessing_root.total_covid_scalars(columns=columns))

    def save_seroprevalence(self, data: pd.DataFrame, draw_id: int = None) -> None:
        if draw_id is None:
            key = self.preprocessing_root.seroprevalence()
        else:
            key = self.preprocessing_root.seroprevalence_samples(draw_id=draw_id)
        io.dump(data, key)

    def load_seroprevalence(self, draw_id: int = None) -> pd.DataFrame:
        if draw_id is None:
            key = self.preprocessing_root.seroprevalence()
        else:
            key = self.preprocessing_root.seroprevalence_samples(draw_id=draw_id)
        return io.load(key)

    def save_sensitivity(self, data: pd.DataFrame, draw_id: int = None) -> None:
        if draw_id is None:
            key = self.preprocessing_root.sensitivity()
        else:
            key = self.preprocessing_root.sensitivity_samples(draw_id=draw_id)
        io.dump(data, key)

    def load_sensitivity(self, draw_id: int = None) -> pd.DataFrame:
        if draw_id is None:
            key = self.preprocessing_root.sensitivity()
        else:
            key = self.preprocessing_root.sensitivity_samples(draw_id=draw_id)
        return io.load(key)

    def save_testing_data(self, data: pd.DataFrame):
        io.dump(data, self.preprocessing_root.testing_for_idr())

    def load_testing_data(self):
        return io.load(self.preprocessing_root.testing_for_idr())

    def save_covariate(self, data: pd.DataFrame, covariate: str, scenario: str) -> None:
        io.dump(data, self.preprocessing_root[covariate](covariate_scenario=scenario))

    def load_covariate(self, covariate: str, scenario: str) -> pd.DataFrame:
        return io.load(self.preprocessing_root[covariate](covariate_scenario=scenario))

    def save_covariate_info(self, data: pd.DataFrame, covariate: str, info_type: str) -> None:
        io.dump(data, self.preprocessing_root[f"{covariate}_info"](info_type=info_type))

    def load_covariate_info(self, covariate: str, info_type: str) -> pd.DataFrame:
        return io.load(self.preprocessing_root[f"{covariate}_info"](info_type=info_type))

    def save_variant_prevalence(self, data: pd.DataFrame, scenario: str) -> None:
        io.dump(data, self.preprocessing_root.variant_prevalence(scenario=scenario))

    def load_variant_prevalence(self, scenario: str) -> pd.DataFrame:
        return io.load(self.preprocessing_root.variant_prevalence(scenario=scenario))

    def save_waning_parameters(self, data: pd.DataFrame, measure: str) -> None:
        io.dump(data, self.preprocessing_root.waning_parameters(measure=measure))

    def load_waning_parameters(self, measure: str) -> pd.DataFrame:
        return io.load(self.preprocessing_root.waning_parameters(measure=measure))

    def save_vaccine_uptake(self, data: pd.DataFrame, scenario: str) -> None:
        io.dump(data, self.preprocessing_root.vaccine_uptake(covariate_scenario=scenario))

    def load_vaccine_uptake(self, scenario: str) -> pd.DataFrame:
        return io.load(self.preprocessing_root.vaccine_uptake(covariate_scenario=scenario))

    def save_vaccine_risk_reduction(self, data: pd.DataFrame, scenario: str) -> None:
        io.dump(data, self.preprocessing_root.vaccine_risk_reduction(covariate_scenario=scenario))

    def load_vaccine_risk_reduction(self, scenario: str) -> pd.DataFrame:
        return io.load(self.preprocessing_root.vaccine_risk_reduction(covariate_scenario=scenario))
