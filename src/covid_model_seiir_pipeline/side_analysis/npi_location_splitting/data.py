from pathlib import Path
from typing import List
import re

import pandas as pd

from covid_model_seiir_pipeline.lib import cli_tools


class DataLoader:
    def __init__(self,
                 model_inputs_version: str,
                 seir_outputs_version: str,
                 write: bool,):
        # input case/death data for split locations
        self.nyt_data_path = Path('/mnt/share/covid-19/side-analyses/npi-analysis/covariates/nyt_data'
                                  '/nyt_county_cases_deaths.csv')
        self.full_data_path = Path(f'/mnt/share/covid-19/model-inputs/{model_inputs_version}/full_data_unscaled.csv')
        self.bra_data_path = Path('/mnt/share/covid-19/side-analyses/npi-analysis/covariates/brazil_admin2'
                                  '/brazil_admin2_cases_deaths.csv')
        self.pre_fix_path_str = (f'/mnt/share/covid-19/model-inputs/{model_inputs_version}/raw_formatted'
                                 '/intermediate_data_fixes/{measure}_0_pre_data_fixes.csv')

        # metadata
        self.hierarchy_path = Path('/mnt/share/covid-19/model-inputs/latest/locations/npi_hierarchy.csv')
        self.ihme_population_path = Path('/mnt/share/covid-19/model-inputs/latest/output_measures'
                                         '/population/all_populations.csv')
        self.supp_population_path = Path('/mnt/share/covid-19/side-analyses/npi-analysis/covariates'
                                         '/populations/brazil_city_us_county_populations.csv')

        # infections estimates for model locations
        self.covid_model_path = Path(f'/mnt/share/covid-19/seir-outputs/{seir_outputs_version}/reference/output_summaries'
                                     '/daily_infections.csv')

        # where to find shapefiles
        self.shapefile_root = Path('/snfs1/WORK/11_geospatial/admin_shapefiles/2022_05_09')

        # create run directory in output root for final outputs
        output_root = Path('/mnt/share/covid-19/side-analyses/npi-analysis/infections')
        if write:
            cli_tools.setup_directory_structure(output_root, with_production=False)
            self.run_directory = cli_tools.make_run_directory(output_root)
        else:
            self.run_directory = None

    def pre_fix_path(self, measure: str):
        return Path(self.pre_fix_path_str.format(measure=measure))

    def metadata_dict(self):
        return dict(
            hierarchy_path=self.hierarchy_path,
            covid_model_path=self.covid_model_path,
            nyt_data_path=self.nyt_data_path,
            full_data_path=self.full_data_path,
            bra_data_path=self.bra_data_path,
            pre_fix_path_str=self.pre_fix_path_str,
        )

    def load_hierarchy(self) -> pd.DataFrame:
        hierarchy = pd.read_csv(self.hierarchy_path)

        return hierarchy

    @staticmethod
    def __load_cumulative_measure_data(path: Path, measure: str, location_ids: List[int]) -> pd.Series:
        data = pd.read_csv(path, encoding='latin1')

        data = data.loc[data['location_id'].isin(location_ids)].reset_index(drop=True)

        for old_name, new_name in [('Date', 'date'),
                                   ('Confirmed', 'cases'),
                                   ('Deaths', 'deaths'),
                                   ('value', measure)]:
            if old_name in data:
                data = data.rename(columns={old_name: new_name})

        alt_date_format = re.compile(r'\d\d.\d\d.\d\d\d\d')
        if alt_date_format.match(data['date'][0]):
            data['date'] = pd.to_datetime(data['date'], format='%d.%m.%Y')
        else:
            data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

        data = data.set_index(['location_id', 'date']).sort_index().loc[:, measure]

        if str(path).startswith('/mnt/share/covid-19/model-inputs/') and str(path).endswith('full_data_unscaled.csv'):
            # just take Spokane from NYT data like all the other counties
            data = data.drop(3539)

        drop_location_ids = [
            7,    # Democratic People's Republic of Korea
            23,   # Kiribati
            24,   # Marshall Islands
            25,   # Micronesia (Federated States of)
            27,   # Samoa
            28,   # Solomon Islands
            29,   # Tonga
            30,   # Vanuatu
            39,   # Tajikistan
            66,   # Brunei Darussalam
            131,  # Nicaragua
            175,  # Burundi
            176,  # Comoros
            177,  # Djibouti
            183,  # Mauritius
            186,  # Seychelles
            189,  # United Republic of Tanzania
            215,  # Sao Tome and Principe
            298,  # American Samoa
            349,  # Greenland
            369,  # Nauru
            376,  # Northern Mariana Islands
            380,  # Palau
            416,  # Tuvalu
        ]
        data = data.drop(drop_location_ids, errors='ignore')

        return data

    def load_raw_data(self, which_locs: str, measure: str, hierarchy: pd.DataFrame) -> pd.Series:
        if which_locs == 'all':
            raw_paths = [self.nyt_data_path,
                         self.full_data_path,
                         self.bra_data_path,]
            location_ids = hierarchy.loc[hierarchy['most_detailed'] == 1, 'location_id'].to_list()
        elif which_locs == 'bra_admin1':
            raw_paths = [self.pre_fix_path(measure)]
            location_ids = hierarchy.loc[hierarchy['parent_id'] == 135, 'location_id'].to_list()
        elif which_locs == 'bra_admin2':
            raw_paths = [self.bra_data_path]
            location_ids = hierarchy.loc[hierarchy['most_detailed'] == 1, 'location_id'].to_list()

        return pd.concat([
            self.__load_cumulative_measure_data(raw_path, measure, location_ids)
            for raw_path in raw_paths
        ])

    def load_populations(self) -> pd.Series:
        ihme_populations = pd.read_csv(self.ihme_population_path)
        ihme_populations = ihme_populations.loc[
            (ihme_populations['year_id'] == 2019)
            & (ihme_populations['age_group_id'] == 22)
            & (ihme_populations['sex_id'] == 3)
        ]
        ihme_populations = ihme_populations.set_index('location_id').loc[:, 'population']

        supp_populations = pd.read_csv(self.supp_population_path)
        supp_populations = supp_populations.set_index('location_id').loc[:, 'population']

        populations = pd.concat(
            [
                supp_populations.drop(ihme_populations.index, errors='ignore'),
                ihme_populations
            ]
        )
        return populations.sort_index()

    def load_infections_estimates(self) -> pd.Series:
        data = pd.read_csv(self.covid_model_path)

        data['date'] = pd.to_datetime(data['date'])
        data = data.rename(columns={'location_id': 'model_location_id'})

        data = data.set_index(['model_location_id', 'date']).loc[:, 'mean'].rename('daily_infections')

        return data
