from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from covid_model_seiir_pipeline.lib import (
    io
)
from covid_model_seiir_pipeline.pipeline.forecasting import (
    ForecastDataInterface,
    ForecastSpecification,
)
from covid_model_seiir_pipeline.side_analysis.counterfactual.specification import (
    CounterfactualSpecification,
)


class CounterfactualDataInterface:

    def __init__(self,
                 forecast_data_interface: ForecastDataInterface,
                 input_root: io.CounterfactualInputRoot,
                 output_root: io.CounterfactualRoot):
        self.forecast_data_interface = forecast_data_interface
        self.input_root = input_root
        self.output_root = output_root

    @classmethod
    def from_specification(cls, specification: CounterfactualSpecification) -> 'CounterfactualDataInterface':
        forecast_spec = ForecastSpecification.from_version_root(specification.data.seir_forecast_version)
        forecast_data_interface = ForecastDataInterface.from_specification(forecast_spec)
        return cls(
            forecast_data_interface=forecast_data_interface,
            input_root=io.CounterfactualInputRoot(
                specification.data.counterfactual_input_version, data_format='parquet'
            ),
            output_root=io.CounterfactualRoot(
                specification.data.output_root, data_format=specification.data.output_format
            ),
        )

    def make_dirs(self, **prefix_args):
        io.touch(self.output_root, **prefix_args)

    def load_location_ids(self):
        return self.forecast_data_interface.load_location_ids()

    def load_past_compartments(self, draw_id: int):
        return self.forecast_data_interface.load_past_compartments(draw_id)
