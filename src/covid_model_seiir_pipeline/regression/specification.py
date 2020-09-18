from dataclasses import dataclass, field
from typing import Dict, Tuple, List

from covid_model_seiir_pipeline.utilities import Specification, asdict
from covid_model_seiir_pipeline.workflow_template import (
    ExecutorParameters,
    WorkflowSpecification,
    DEFAULT_PROJECT,
    DEFAULT_QUEUE
)


class RegressionTaskParams(ExecutorParameters):

    @property
    def default_cores(self) -> int:
        return 1

    @property
    def default_memory(self) -> str:
        return '2G'

    @property
    def default_runtime(self) -> int:
        return 3000


class RegressionWorkflowSpecification(WorkflowSpecification):

    @classmethod
    def from_dict(cls, regression_workflow_spec: Dict):
        project = regression_workflow_spec.get('project', DEFAULT_PROJECT)
        queue = regression_workflow_spec.get('queue', DEFAULT_QUEUE)
        regression_task_spec = regression_workflow_spec.get('beta_regression', {})
        return cls(project, {'beta_regression': RegressionTaskParams(queue, **regression_task_spec)})


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
        return asdict(self)


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
                 parameters: FitParameters,
                 covariates: List[CovariateSpecification],
                 workflow: RegressionWorkflowSpecification):
        self._data = data
        self._parameters = parameters
        self._covariates = {c.name: c for c in covariates}
        self._workflow = workflow

    @classmethod
    def parse_spec_dict(cls, regression_spec_dict: Dict) -> Tuple:
        """Constructs a regression specification from a dictionary."""
        data = RegressionData(**regression_spec_dict.get('data', {}))
        parameters = FitParameters(**regression_spec_dict.get('parameters', {}))

        # covariates
        cov_dicts = regression_spec_dict.get('covariates', {})
        covariates = []
        for name, cov_spec in cov_dicts.items():
            covariates.append(CovariateSpecification(name, **cov_spec))
        if not covariates:  # Assume we're generating for a template
            covariates.append(CovariateSpecification())

        workflow = RegressionWorkflowSpecification.from_dict(regression_spec_dict.get('workflow', {}))

        return data, parameters, covariates, workflow

    @property
    def data(self) -> RegressionData:
        """The data specification for the regression."""
        return self._data

    @property
    def parameters(self) -> FitParameters:
        """The parametrization of the regression."""
        return self._parameters

    @property
    def covariates(self) -> Dict[str, CovariateSpecification]:
        """The covariates for the regression."""
        return self._covariates

    @property
    def workflow(self) -> RegressionWorkflowSpecification:
        """The regression workflow specification"""
        return self._workflow

    def to_dict(self) -> Dict:
        """Converts the specification to a dict."""
        spec = {
            'data': self.data.to_dict(),
            'parameters': self.parameters.to_dict(),
            'covariates': {k: v.to_dict() for k, v in self._covariates.items()},
            'workflow': self.workflow.to_dict(),
        }
        return spec

