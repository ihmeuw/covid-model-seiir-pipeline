from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Tuple

from covid_model_seiir_pipeline.lib import (
    utilities,
    workflow,
)


class __RegressionJobs(NamedTuple):
    regression: str = 'regression'


REGRESSION_JOBS = __RegressionJobs()


class RegressionTaskSpecification(workflow.TaskSpecification):
    """Specification of execution parameters for regression tasks."""
    default_max_runtime_seconds = 3000
    default_m_mem_free = '2G'
    default_num_cores = 1


class RegressionWorkflowSpecification(workflow.WorkflowSpecification):
    """Specification of execution parameters for regression workflows."""
    tasks = {
        REGRESSION_JOBS.regression: RegressionTaskSpecification,
    }


@dataclass
class RegressionData:
    """Specifies the inputs and outputs for a regression"""
    covariate_version: str = field(default='best')
    infection_version: str = field(default='best')
    location_set_version_id: int = field(default=0)
    location_set_file: str = field(default='')
    output_root: str = field(default='')

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        return utilities.asdict(self)


@dataclass
class FitParameters:
    """Specifies the parameters of the ODE fit."""
    n_draws: int = field(default=1000)

    day_shift: Tuple[int, int] = field(default=(0, 8))

    alpha: Tuple[float, float] = field(default=(0.9, 1.0))
    sigma: Tuple[float, float] = field(default=(0.2, 1/3))
    gamma1: Tuple[float, float] = field(default=(0.5, 0.5))
    gamma2: Tuple[float, float] = field(default=(1/3, 1.0))
    solver_dt: float = field(default=0.1)
    sequential_refit: bool = field(default=False)

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        return utilities.asdict(self)


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
        return {k: v for k, v in utilities.asdict(self).items() if k != 'name'}


class RegressionSpecification(utilities.Specification):
    """Specification for a regression run."""

    def __init__(self,
                 data: RegressionData,
                 parameters: FitParameters,
                 covariates: List[CovariateSpecification],
                 workflow: RegressionWorkflowSpecification):
        self._data = data
        self._workflow = workflow
        self._parameters = parameters
        self._covariates = {c.name: c for c in covariates}

    @classmethod
    def parse_spec_dict(cls, regression_spec_dict: Dict) -> Tuple:
        """Constructs a regression specification from a dictionary."""
        data = RegressionData(**regression_spec_dict.get('data', {}))
        parameters = FitParameters(**regression_spec_dict.get('parameters', {}))
        workflow = RegressionWorkflowSpecification(**regression_spec_dict.get('workflow', {}))

        # covariates
        cov_dicts = regression_spec_dict.get('covariates', {})
        covariates = []
        for name, cov_spec in cov_dicts.items():
            covariates.append(CovariateSpecification(name, **cov_spec))
        if not covariates:  # Assume we're generating for a template
            covariates.append(CovariateSpecification())

        return data, parameters, covariates, workflow

    @property
    def data(self) -> RegressionData:
        """The data specification for the regression."""
        return self._data

    @property
    def workflow(self) -> RegressionWorkflowSpecification:
        """The workflow specification for the regression."""
        return self._workflow

    @property
    def parameters(self) -> FitParameters:
        """The parametrization of the regression."""
        return self._parameters

    @property
    def covariates(self) -> Dict[str, CovariateSpecification]:
        """The covariates for the regression."""
        return self._covariates

    def to_dict(self) -> Dict:
        """Converts the specification to a dict."""
        spec = {
            'data': self.data.to_dict(),
            'workflow': self.workflow.to_dict(),
            'parameters': self.parameters.to_dict(),
            'covariates': {k: v.to_dict() for k, v in self._covariates.items()},
        }
        return spec

