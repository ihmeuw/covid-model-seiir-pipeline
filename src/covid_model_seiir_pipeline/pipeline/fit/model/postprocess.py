import functools
import multiprocessing
from typing import Any, Callable, List, Optional

import pandas as pd
import tqdm

from covid_model_seiir_pipeline.pipeline.fit.data import (
    FitDataInterface,
)


class MeasureConfig:
    def __init__(self,
                 loader: Callable[['FitDataInterface', pd.Index, int], pd.DataFrame],
                 label: str,
                 cumulative_label: str = None,
                 aggregator: Callable = None):
        self.loader = loader
        self.label = label
        self.cumulative_label = cumulative_label
        self.aggregator = aggregator


MEASURES = {
}
