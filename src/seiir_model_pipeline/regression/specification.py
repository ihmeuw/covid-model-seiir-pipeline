from dataclasses import dataclass, field
import itertools
from string import Formatter
from typing import Dict, Tuple, List, Optional, Dict

from seiir_model_pipeline.utilities import Specification, asdict


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

    # i/o params
    name: str = field(default='covariate')
    scenario: str = field(default='reference')
    input_file_pattern: str = field(default="{name}_{scenario}")
    output_file_pattern: str = field(default="{name}_{scenario}")
    alternate_scenarios: Optional[List[str]] = field(default=None)

    # model params
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

    def get_input_file(self, scenario: str):
        input_file_keys = [i[1] for i in Formatter().parse(self.input_file_pattern)
                           if i[1] is not None]

        format_spec: Dict[str, str] = {}
        if "name" in input_file_keys:
            format_spec["name"] = self.name
        if "scenario" in input_file_keys:
            format_spec["scenario"] = scenario
        return self.input_file_pattern.format(**format_spec)

    def get_output_file(self, scenario: str):
        output_file_keys = [i[1] for i in Formatter().parse(self.output_file_pattern)
                            if i[1] is not None]

        format_spec: Dict[str, str] = {}
        if "name" in output_file_keys:
            format_spec["name"] = self.name
        if "scenario" in output_file_keys:
            format_spec["scenario"] = scenario
        return self.output_file_pattern.format(**format_spec)


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
