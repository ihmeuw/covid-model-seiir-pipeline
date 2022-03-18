from pathlib import Path
from typing import Dict, List, Optional, Union

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

    def get_n_oversample_draws(self) -> int:
        specification = self.load_specification()
        return specification.data.n_oversample_draws

    def get_n_total_draws(self):
        return self.get_n_draws() + self.get_n_oversample_draws()

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
        # FIXME: Something super weird going on where the data is not the same.  Maybe set & sort index?
        path = self.model_inputs_root._root / 'serology' / 'global_serology_summary.csv'
        data = pd.read_csv(path)
        return data

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
        data = data.reset_index().rename(columns=col_map)
        data['date'] = pd.to_datetime(data['Date'])
        data['location_id'] = data['location_id'].astype(int)
        data = (data
                .set_index(['location_id', 'date'])
                .sort_index()
                .loc[:, col_map.values()])
        return data

    def load_hospital_census_data(self) -> pd.DataFrame:
        # Locations to exclude from the census input data. Requested by Steve on 9/30/2020
        # TODO: Revisit this list, it's very old.
        census_exclude_locs = [
            26,   # Papua New Guinea
            58,   # Estonia
            67,   # Japan
            69,   # Singapore
            74,   # Andorra
            144,  # Jordan
            170,  # Congo
            172,  # Equatorial Guinea
            179,  # Ethiopia
            200,  # Benin
        ]
        corrections_data = {
            'hospital_census': io.load(self.model_inputs_root.hospital_census()),
            'icu_census': io.load(self.model_inputs_root.icu_census()),
        }
        for measure, data in corrections_data.items():
            data = data.reset_index()
            all_age_sex = (data.age_group_id == 22) & (data.sex_id == 3)
            drop_locs = data.location_id.isin(census_exclude_locs)
            keep_cols = ["location_id", "date", "value"]
            data = data.loc[all_age_sex & ~drop_locs, keep_cols]
            corrections_data[measure] = data.set_index(['location_id', 'date']).sort_index().value.rename(measure)
        return pd.concat(corrections_data.values(), axis=1)

    def load_hospital_bed_capacity(self) -> pd.DataFrame:
        return io.load(self.model_inputs_root.hospital_capacity())

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
            data.iloc[:, 1] += 1
            # Change age group end of the terminal group from 99 to 125
            data.iloc[-1, 1] = 125
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
                'relaxed': 'mask_use_relaxed',
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
                'reference': 'last_shots_in_arm_by_brand_w_booster_reference',
                'booster': 'last_shots_in_arm_by_brand_w_booster_optimal',
                'probably_no': 'last_shots_in_arm_by_brand_w_booster_no_prob',
            }[scenario]
        except KeyError:
            raise ValueError(f'Unknown vaccine scenario {scenario}.')
        data = io.load(self.vaccine_coverage_root.brand_specific_coverage(measure=scenario_file))
        data = data.drop(columns='Unnamed: 0').reset_index()
        return data

    def load_serology_vaccine_coverage(self) -> pd.DataFrame:
        summary_data = io.load(self.serology_vaccine_coverage_root.old_vaccine_coverage())

        keep_columns = [
            # total vaccinated (all and by three groups)
            'cumulative_all_effective',
            'cumulative_all_vaccinated',
            'cumulative_all_fully_vaccinated',
            'cumulative_essential_vaccinated',
            'cumulative_adults_vaccinated',
            'cumulative_elderly_vaccinated',
            'hr_vaccinated',
            'lr_vaccinated',
        ]
        summary_data = summary_data.sort_index().loc[:, keep_columns]

        series_hesitancy = io.load(self.serology_vaccine_coverage_root.series_hesitancy())
        series_hesitancy = (series_hesitancy
                            .sort_index()
                            .smooth_combined_yes
                            .rename('vaccine_acceptance'))
        point_hesitancy = io.load(self.serology_vaccine_coverage_root.point_hesitancy())
        point_hesitancy = (point_hesitancy
                           .sort_index()
                           .smooth_combined_yes
                           .rename('vaccine_acceptance_point')
                           .reindex(series_hesitancy.index, level='location_id'))
        data = pd.concat([summary_data, series_hesitancy, point_hesitancy], axis=1)
        return data

    #########################
    # Vaccine efficacy data #
    #########################

    def load_efficacy_table(self) -> pd.DataFrame:
        data = io.load(self.vaccine_efficacy_root.efficacy_table())
        data = (data
                .rename(columns={'merge_name': 'brand'})
                .set_index(['brand', 'vaccine_course'])
                .loc[:, ['efficacy', 'prop_protected_not_infectious',
                         'variant_efficacy', 'variant_infection',
                         'omicron_efficacy', 'omicron_infection']])
        # Use more sensible column names that care about variants and efficacy endpoints.
        data.columns = pd.MultiIndex.from_tuples(
            [('ancestral', 'severe_disease'),
             ('ancestral', 'infection'),
             ('delta', 'severe_disease'),
             ('delta', 'infection'),
             ('omicron', 'severe_disease'),
             ('omicron', 'infection')],
            names=[None, 'endpoint']
        )
        # Pivot endpoint from the columns to the index.
        data = (data
                .stack()
                .reorder_levels(['endpoint', 'brand', 'vaccine_course'])
                .sort_index())
        return data

    def load_waning_data(self) -> pd.DataFrame:
        data = io.load(self.vaccine_efficacy_root.waning_distribution(measure='vetsvacc'))
        expected_cols = {
            'mid_point',
            'logit_pfi_inf', 'pfi_inf',
            'logit_mod_inf', 'mod_inf',
            'logit_ast_inf', 'ast_inf',
            'logit_jan_inf', 'jan_inf',
            'logit_pfi_sev', 'pfi_sev',
            'logit_mod_sev', 'mod_sev',
            'logit_ast_sev', 'ast_sev',
            'jan_sev',
        }
        assert set(data.columns) == expected_cols
        data = (data
                .rename(columns={'mid_point': 'weeks_since_delivery'})
                .set_index('weeks_since_delivery'))

        endpoint_map = {'inf': 'infection', 'sev': 'severe_disease',}

        out = []
        for col in data.columns:
            brand_endpoint = col.split('_')
            if len(brand_endpoint) == 3:
                continue
            brand, endpoint = brand_endpoint
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
        return io.load(self.preprocessing_root.hierarchy(measure=name)).reset_index()

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
        if draw_id is not None:
            data = io.load(self.preprocessing_root.total_covid_scalars(columns=[f'draw_{draw_id}']))
            data = data.rename(columns={f'draw_{draw_id}': 'scalar'})
        else:
            data = io.load(self.preprocessing_root.total_covid_scalars())
        return data

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

    def save_vaccine_summary(self, data: pd.DataFrame) -> None:
        io.dump(data, self.preprocessing_root.vaccine_summary())

    def load_vaccine_summary(self, columns: List[str] = None) -> pd.DataFrame:
        return io.load(self.preprocessing_root.vaccine_summary(columns=columns))

    def save_vaccine_uptake(self, data: pd.DataFrame, scenario: str) -> None:
        io.dump(data, self.preprocessing_root.vaccine_uptake(covariate_scenario=scenario))

    def load_vaccine_uptake(self, scenario: str) -> pd.DataFrame:
        return io.load(self.preprocessing_root.vaccine_uptake(covariate_scenario=scenario))

    def save_vaccine_risk_reduction(self, data: pd.DataFrame, scenario: str) -> None:
        io.dump(data, self.preprocessing_root.vaccine_risk_reduction(covariate_scenario=scenario))

    def load_vaccine_risk_reduction(self, scenario: str) -> pd.DataFrame:
        return io.load(self.preprocessing_root.vaccine_risk_reduction(covariate_scenario=scenario))
