from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Dict, Union

import yaml

from seiir_model_pipeline.utilities import load_specification, asdict


@dataclass
class ForecastData:
    """Specifies the inputs and outputs for a forecast."""
    regression_version: str = field(default='best')
    output_root: str = field(default='')

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        return asdict(self)


@dataclass
class ScenarioSpecification:
    """Forecasting specification for a scenario."""
    ALLOWED_ALGORITHMS = ('RK45',)
    NO_BETA_SCALING = -1

    name: str = field(default='dummy_scenario')
    covariates: Dict[str, str] = field(default_factory=dict)
    beta_scaling_window: int = field(default=NO_BETA_SCALING)
    algorithm: str = field(default='RK45')

    def __post_init__(self):
        if self.algorithm not in self.ALLOWED_ALGORITHMS:
            raise ValueError(f'Unknown algorithm {self.algorithm} in scenario {self.name}. '
                             f'Allowed algorithms are {self.ALLOWED_ALGORITHMS}.')

        if not isinstance(self.beta_scaling_window, int) or self.beta_scaling_window < -1:
            raise TypeError(f'Beta scaling window must be a positive int or -1 indicating no scaling. '
                            f'Scaling window for scenario {self.name} is {self.beta_scaling_window}.')

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        return asdict(self)


class ForecastSpecification:

    def __init__(self, data: ForecastData, *scenarios: ScenarioSpecification):
        self._data = data
        self._scenarios = {s.name: s for s in scenarios}

    @classmethod
    def from_dict(cls, forecast_spec_dict: Dict) -> 'ForecastSpecification':
        data = ForecastData(**forecast_spec_dict.get('data', {}))
        scenario_dicts = forecast_spec_dict.get('scenarios', {})
        scenarios = []
        for name, scenario_spec in scenario_dicts.items():
            scenarios.append(ScenarioSpecification(name, **scenario_spec))
        if not scenarios:
            scenarios.append(ScenarioSpecification())
        return cls(data, *scenarios)

    @property
    def data(self) -> 'ForecastData':
        return self._data

    @property
    def scenarios(self) -> Dict[str, ScenarioSpecification]:
        return self._scenarios

    def to_dict(self):
        return {
            'data': self.data.to_dict(),
            'scenarios': {k: v.to_dict() for k, v in self._scenarios.items()}
        }

    def __repr__(self):
        return f'ForecastSpecification(\n{pformat(self.to_dict())}\n)'



def load_forecast_specification(specification_path: Union[str, Path]) -> ForecastSpecification:
    """Loads a forecast specification from a yaml file."""
    spec_dict = load_specification(specification_path)
    return ForecastSpecification.from_dict(spec_dict)


def dump_forecast_specification(forecast_specification: ForecastSpecification,
                                specification_path: Union[str, Path]) -> None:
    """Writes a forecast specification to a yaml file."""
    with Path(specification_path).open('w') as specification_file:
        yaml.dump(forecast_specification.to_dict(), specification_file, sort_keys=False)


def validate_specification(forecast_specification: ForecastSpecification) -> None:
    """Checks all preconditions on the forecast."""
    pass
