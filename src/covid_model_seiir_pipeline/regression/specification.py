from dataclasses import dataclass, field
from typing import Dict, Tuple, List

from covid_model_seiir_pipeline.utilities import Specification, asdict


@dataclass
class RegressionData:
    """Specifies the inputs and outputs for a regression"""
    covariate_version: str = field(default='best')
    ode_fit_version: str = field(default='best')
    output_root: str = field(default='')

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        return asdict(self)


@dataclass
class CovariateSpecification:
    """Regression specification for a covariate."""

    # model params
    name: str = field(default='covariate')
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


class RegressionSpecification(Specification):
    """Specification for a regression run."""

    def __init__(self,
                 data: RegressionData,
                 covariates: List[CovariateSpecification]):
        self._data = data
        self._covariates = {c.name: c for c in covariates}

    @classmethod
    def parse_spec_dict(cls, regression_spec_dict: Dict) -> Tuple:
        """Constructs a regression specification from a dictionary."""
        data = RegressionData(**regression_spec_dict.get('data', {}))

        # covariates
        cov_dicts = regression_spec_dict.get('covariates', {})
        covariates = []
        for name, cov_spec in cov_dicts.items():
            covariates.append(CovariateSpecification(name, **cov_spec))
        if not covariates:  # Assume we're generating for a template
            covariates.append(CovariateSpecification())

        return data, covariates

    @property
    def data(self) -> RegressionData:
        """The data specification for the regression."""
        return self._data

    @property
    def covariates(self) -> Dict[str, CovariateSpecification]:
        """The covariates for the regression."""
        return self._covariates

    def to_dict(self) -> Dict:
        """Converts the specification to a dict."""
        spec = {
            'data': self.data.to_dict(),
            'covariates': {k: v.to_dict() for k, v in self._covariates.items()},
        }
        return spec


def validate_specification(regression_specification: RegressionSpecification) -> None:
    """Checks all preconditions on the regression."""
    pass
