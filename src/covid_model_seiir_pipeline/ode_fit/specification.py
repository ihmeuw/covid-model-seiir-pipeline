from dataclasses import dataclass, field
from typing import Dict, Tuple, Union

from loguru import logger
import numpy as np

from covid_model_seiir_pipeline.utilities import Specification, asdict


@dataclass
class FitData:
    """Specifies the inputs and outputs for an ODE fit."""
    infection_version: str = field(default='best')
    location_set_version_id: int = field(default=0)
    location_set_file: str = field(default='')
    output_root: str = field(default='')

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        return asdict(self)


@dataclass
class FitParameters:
    """Specifies the parameters of the ODE fit."""
    n_draws: int = field(default=1000)

    degree: int = field(default=3)
    knots: Union[Tuple[float], np.array] = field(default=(0.0, 0.25, 0.5, 0.75, 1.0))
    concavity: bool = field(default=False)
    increasing: bool = field(default=False)
    spline_se_power: float = field(default=0.2)
    spline_space: str = field(default='ln daily')
    spline_knots_type: str = field(default='domain')
    spline_r_linear: bool = True
    spline_l_linear: bool = True

    day_shift: Tuple[int, int] = field(default=(0, 8))

    alpha: Tuple[float, float] = field(default=(0.9, 1.0))
    sigma: Tuple[float, float] = field(default=(0.2, 1/3))
    gamma1: Tuple[float, float] = field(default=(0.5, 0.5))
    gamma2: Tuple[float, float] = field(default=(1/3, 1.0))
    solver_dt: float = field(default=0.1)

    def __post_init__(self) -> None:
        # Do any necessary type coercion
        self.knots = np.array(self.knots)

        # TODO: Add preconditions for all variables.
        # Check specification preconditions.
        if not self.spline_space.startswith('ln') and self.spline_se_power != 0.0:
            logger.warning("Spline not fitting in the log space, spline_se_power advised to be 0.")

        if self.spline_space.endswith('daily') and self.increasing:
            logger.warning("Spline fitting daily data, do not suggest using increasing constraint.")

        if self.spline_space.endswith('cumul') and self.concavity:
            logger.warning("Spline fitting cumulative data, do not suggest using concave constraint.")

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        return asdict(self)


class FitSpecification(Specification):
    """Specification for an ODE fit run."""

    def __init__(self,
                 data: FitData,
                 parameters: FitParameters):
        self._data = data
        self._parameters = parameters

    @classmethod
    def parse_spec_dict(cls, fit_spec_dict: Dict) -> Tuple:
        """Constructs a fit specification from a dictionary."""
        data = FitData(**fit_spec_dict.get('data', {}))
        parameters = FitParameters(**fit_spec_dict.get('parameters', {}))
        return data, parameters

    @property
    def data(self) -> FitData:
        """The data specification for the regression."""
        return self._data

    @property
    def parameters(self) -> FitParameters:
        """The parameterization of the regression."""
        return self._parameters

    def to_dict(self) -> Dict:
        """Converts the specification to a dict."""
        return {
            'data': self.data.to_dict(),
            'parameters': self.parameters.to_dict()
        }
