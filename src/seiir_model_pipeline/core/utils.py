import pandas as pd
import numpy as np

from seiir_model_pipeline.core.file_master import PEAK_DATE_FILE


def get_peaked_dates_from_file():
    df = pd.read_csv(PEAK_DATE_FILE)
    # convert date to numpy
    df.set_index('location')
    return df.to_dict()


def sample_params_from_bounds():
    return 1.
