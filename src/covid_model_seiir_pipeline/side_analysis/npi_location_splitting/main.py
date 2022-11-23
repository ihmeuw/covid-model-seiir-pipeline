from typing import Tuple
from loguru import logger

import pandas as pd

from covid_shared import shell_tools

from covid_model_seiir_pipeline.side_analysis.npi_location_splitting.data import (
    DataLoader,
)
from covid_model_seiir_pipeline.side_analysis.npi_location_splitting.model import (
    generate_measure_specific_infections,
    combine_measure_specific_infections,
)
from covid_model_seiir_pipeline.side_analysis.npi_location_splitting.diagnostics import (
    country_map,
)


def generate_infections_inputs(data_loader: DataLoader) -> Tuple[pd.DataFrame, pd.Series]:
    logger.info('SETTING UP DATA LOADER')
    hierarchy = data_loader.load_hierarchy()

    logger.info('PROCESSING DATA AND SPLITTING MODEL LOCATION INFECTIONS BY MEASURE')
    raw_data, processed_data, infections = generate_measure_specific_infections(data_loader, hierarchy)

    logger.info('GENERATING SPARSENESS WEIGHTS, EXTRAPOLATING MISSING MEASURES AT TAILS, AND COMBINING INFECTIONS')
    weights, infections = combine_measure_specific_infections(infections, processed_data)

    logger.info('SUBSETTING TO MODEL LOCATIONS AND IDENTIFYING MISSING LOCATIONS')
    infections = infections.loc[
        [location_id for location_id in infections.index.get_level_values('location_id').unique()
         if location_id in hierarchy.loc[hierarchy['most_detailed'] == 1, 'location_id'].to_list()]
    ]
    missing_locations = hierarchy.loc[
        (hierarchy['most_detailed'] == 1) &
        (~hierarchy['location_id'].isin(infections.index.get_level_values('location_id').unique())),
        ['location_id', 'location_name']
    ]
    if not missing_locations.empty:
        logger.warning(f'Infections not present for the following most-detailed locations:\n{missing_locations}')

    if data_loader.run_directory:
        logger.info(f'WRITING OUTPUTS TO {data_loader.run_directory}')
        metadata = data_loader.metadata_dict()
        with open(data_loader.run_directory / 'metadata.yaml', 'w') as file:
            yaml.dump(metadata, file)
        for dataset, filename in [
            (hierarchy, 'hierarchy'),
            (missing_locations, 'missing_locations'),
            (raw_data.reset_index(), 'raw_input_data'),
            (processed_data.reset_index(), 'processed_input_data'),
            (infections.reset_index(), 'daily_infections'),
        ]:
            dataset.to_csv(data_loader.run_directory / f'{filename}.csv', index=False)

    # produce diagnosticss
    if data_loader.run_directory:
        shell_tools.mkdir(data_loader.run_directory / 'diagnostics', exists_ok=True)

    logger.info('PRODUCING DIAGNOSTICS -- MAPS')
    country_location_ids = [
        102,  # United States
        196,  # South Africa
        11,   # Indonesia
    ]
    map_dates = ['2021-11-01', 'last']
    map_args = [(country_location_id, map_date) for country_location_id in country_location_ids for map_date in map_dates]
    _ = [country_map(
         country_location_id=country_location_id,
         data_loader=data_loader,
         map_data=infections.loc[:, 'daily_infections_composite_weighted'].rename('mapvar'),
         hierarchy=hierarchy,
         map_label='Cumulative infections per capita',
         map_date=map_date
     )
     for country_location_id, map_date in map_args
    ]

    return hierarchy, raw_data, processed_data, weights, infections, missing_locations
