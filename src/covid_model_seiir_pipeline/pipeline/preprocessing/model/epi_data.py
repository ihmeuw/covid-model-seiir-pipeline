import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.data import (
    PreprocessingDataInterface,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.model import (
    helpers,
)

logger = cli_tools.task_performance_logger


def preprocess_epi_data(data_interface: PreprocessingDataInterface) -> None:
    logger.info('Loading epi data.', context='read')
    mr_hierarchy = data_interface.load_hierarchy('mr')
    pred_hierarchy = data_interface.load_hierarchy('pred')
    age_pattern_data = data_interface.load_age_pattern_data()
    total_covid_scalars = data_interface.load_raw_total_covid_scalars()
    #cases, hospitalizations, deaths = data_interface.load_epi_measures()

    logger.info('Processing epi data.', context='transform')
    age_pattern_data = _process_age_pattern_data(age_pattern_data, pred_hierarchy)
    total_covid_scalars = _process_scalars(total_covid_scalars, pred_hierarchy)
    global_serology = _process_serology_data(global_serology)

    logger.info('Writing epi data.', context='write')
    data_interface.save_age_patterns(age_pattern_data)
    data_interface.save_total_covid_scalars(total_covid_scalars)
    data_interface.save_global_serology(global_serology)


def _process_age_pattern_data(data: pd.DataFrame, hierarchy: pd.DataFrame):
    # Broadcast over location id.
    data['key'] = 1
    hierarchy = hierarchy.reset_index()
    hierarchy['key'] = 1

    data = (hierarchy
            .loc[:, ['location_id', 'key']]
            .merge(data)
            .set_index(['location_id', 'age_group_years_start', 'age_group_years_end'])
            .sort_index()
            .drop(columns='key'))
    return data


def _process_scalars(data: pd.DataFrame, hierarchy: pd.DataFrame):
    missing_locations = hierarchy.index.difference(data.index).tolist()
    if missing_locations:
        logger.warning(f"Missing scalars for the following locations: {missing_locations}.  Filling with 1.")
    data = data.reindex(hierarchy.index, fill_value=1.)
    return data





def _process_deaths():
    pass
