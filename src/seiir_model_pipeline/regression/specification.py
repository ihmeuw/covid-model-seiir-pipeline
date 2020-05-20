from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Dict, List, Union, Tuple

import numpy as np
import yaml

from seiir_model_pipeline.utilities import load_specification, asdict


@dataclass
class RegressionData:
    """Specifies the inputs and outputs for a regression"""
    covariate_version: str = field(default='best')
    infection_version: str = field(default='best')
    location_set_version_id: int = field(default=0)
    output_root: str = field(default='')

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        return asdict(self)


@dataclass
class RegressionParameters:
    """Specifies the parameterization of the regression."""
    n_draws: int = field(default=1000)
    degree: int = field(default=3)
    knots: Union[Tuple[float], np.array] = field(default=(0.0, 0.25, 0.5, 0.75, 1.0))
    day_shift: Tuple[int, int] = field(default=(0, 8))

    alpha: Tuple[float, float] = field(default=(0.9, 1.0))
    sigma: Tuple[float, float] = field(default=(0.2, 1/3))
    gamma1: Tuple[float, float] = field(default=(0.5, 0.5))
    gamma2: Tuple[float, float] = field(default=(1/3, 1.0))
    solver_dt: float = field(default=0.1)

    def __post_init__(self) -> None:
        self.knots = np.array(self.knots)

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        return asdict(self)


@dataclass
class CovariateSpecification:
    """Regression specification for a covariate."""
    name: str = field(default='dummy_covariate')
    order: int = field(default=0)
    use_re: bool = field(default=False)
    gprior: Tuple[float, float] = field(default=(0., 1000.))
    bounds: Tuple[float, float] = field(default=(-1000., 1000.))
    re_var: float = field(default=1.)
    draws: bool = field(default=False)

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists.

        Drops the name parameter as it's used as a key in the specification.

        """
        return {k: v for k, v in asdict(self).items() if k != 'name'}


class RegressionSpecification:
    """Specification for a regression run."""

    def __init__(self,
                 data: RegressionData,
                 parameters: RegressionParameters,
                 *covariates: CovariateSpecification):
        self._data = data
        self._parameters = parameters
        self._covariates = {c.name: c for c in covariates}

    @classmethod
    def from_dict(cls, regression_spec_dict: Dict) -> 'RegressionSpecification':
        """Constructs a regression specification from a dictionary."""
        data = RegressionData(**regression_spec_dict.get('data', {}))
        parameters = RegressionParameters(**regression_spec_dict.get('parameters', {}))
        cov_dicts = regression_spec_dict.get('covariates', {})
        covariates = []
        for name, cov_spec in cov_dicts.items():
            covariates.append(CovariateSpecification(name, **cov_spec))
        if not covariates:  # Assume we're generating for a template
            covariates.append(CovariateSpecification())
        return cls(data, parameters, *covariates)

    @property
    def data(self) -> RegressionData:
        """The data specification for the regression."""
        return self._data

    @property
    def parameters(self) -> RegressionParameters:
        """The parameterization of the regression."""
        return self._parameters

    @property
    def covariates(self) -> Dict[str, CovariateSpecification]:
        """The covariates for the regression."""
        return self._covariates

    def to_dict(self) -> Dict:
        """Converts the specification to a dict."""
        return {
            'data': self.data.to_dict(),
            'parameters': self.parameters.to_dict(),
            'covariates': {k: v.to_dict() for k, v in self._covariates.items()}
        }

    def __repr__(self):
        return f'RegressionSpecification(\n{pformat(self.to_dict())}\n)'


def load_regression_specification(specification_path: Union[str, Path]) -> RegressionSpecification:
    """Loads a regression specification from a yaml file."""
    spec_dict = load_specification(specification_path)
    return RegressionSpecification.from_dict(spec_dict)


def dump_regression_specification(regression_specification: RegressionSpecification,
                                  specification_path: Union[str, Path]) -> None:
    """Writes a regression specification to a yaml file."""
    with Path(specification_path).open('w') as specification_file:
        yaml.dump(regression_specification.to_dict(), specification_file, sort_keys=False)


def validate_specification(regression_specification: RegressionSpecification) -> None:
    """Checks all preconditions on the regression."""
    pass
