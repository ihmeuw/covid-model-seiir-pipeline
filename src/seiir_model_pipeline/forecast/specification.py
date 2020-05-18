from pathlib import Path
from typing import Dict, Union

import yaml

from seiir_model_pipeline.utilities import load_specification


class ForecastSpecification:

    def to_dict(self):
        return {}

    @classmethod
    def from_dict(cls, forecast_dict: Dict):
        return ForecastSpecification()


def load_forecast_specification(specification_path: Union[str, Path]) -> ForecastSpecification:
    """Loads a forecast specification from a yaml file."""
    spec_dict = load_specification(specification_path)
    return ForecastSpecification.from_dict(spec_dict)


def dump_regression_specification(forecast_specification: ForecastSpecification,
                                  specification_path: Union[str, Path]) -> None:
    """Writes a forecast specification to a yaml file."""
    with Path(specification_path).open('w') as specification_file:
        yaml.dump(forecast_specification.to_dict(), specification_file, sort_keys=False)


def validate_specification(forecast_specification: ForecastSpecification) -> None:
    """Checks all preconditions on the forecast."""
    pass
