import pandas as pd
import numpy as np

from db_queries import get_location_metadata

from seiir_model_pipeline.core.file_master import PEAK_DATE_FILE


def get_peaked_dates_from_file():
    df = pd.read_csv(PEAK_DATE_FILE)
    # convert date to numpy
    df.set_index('location')
    return df.to_dict()


def sample_params_from_bounds(bounds):
    return 1.


def get_cov_model_set_from_settings(cov_settings):
    return None


def get_locations(location_set_version_id):
    df = get_location_metadata(location_set_version_id=location_set_version_id)
    return df.location_id.unique()
