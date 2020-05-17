from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union, Tuple

import numpy as np

from seiir_model_pipeline.utilities import load_specification, asdict


@dataclass
class RegressionData:
    """Specifies the inputs and outputs for a regression"""
    covariate_version: str = field(default='best')
    infection_version: str = field(default='best')
    location_set_version_id: int = field(default=0)
    output_root: str = field(default='')

    def to_dict(self):
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

    def __post_init__(self):
        self.knots = np.array(self.knots)

    def to_dict(self):
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

    def to_dict(self):
        return asdict(self)


class RegressionSpecification:

    def __init__(self,
                 data: RegressionData,
                 parameters: RegressionParameters,
                 *covariates: CovariateSpecification):
        self._data = data
        self._parameters = parameters
        self._covariates = {c.name: c for c in covariates}

    @classmethod
    def from_dict(cls, regression_spec_dict: Dict):
        data = RegressionData(**regression_spec_dict.get('data', {}))
        parameters = RegressionParameters(regression_spec_dict.get('parameters', {}))
        cov_dicts = regression_spec_dict.get('covariates', {})
        covariates = []
        for name, cov_spec in cov_dicts.items():
            covariates.append(CovariateSpecification(name, **cov_spec))
        if not covariates:  # Assume we're generating for a template
            covariates.append(CovariateSpecification())
        return cls(data, parameters, *covariates)

    @property
    def data(self) -> Dict:
        return self._data.to_dict()

    @property
    def parameters(self) -> Dict:
        return self._parameters.to_dict()

    @property
    def covariates(self) -> Dict:
        return {k: v.to_dict() for k, v in self._covariates.items()}

    def to_dict(self) -> Dict:
        return {
            'data': self.data,
            'parameters': self.parameters,
            'covariates': self.covariates
        }


def load_regression_specification(specification_path: Union[str, Path]) -> RegressionSpecification:
    spec_dict = load_specification(specification_path)
    return RegressionSpecification.from_dict(spec_dict)

