from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Tuple, Union

from covid_shared import workflow

from covid_model_seiir_pipeline.lib import (
    utilities,
)


class __ODEFitJobs(NamedTuple):
    ode_fit: str = 'ode_fit'
    synthesis_spline: str = 'synthesis_spline'


ODE_FIT_JOBS = __ODEFitJobs(*__ODEFitJobs._fields)


class ODEFitTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 3000
    default_m_mem_free = '20G'
    default_num_cores = 5


class SynthesisSplineTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 6000
    default_m_mem_free = '100G'
    default_num_cores = 26


class ODEFitWorkflowSpecification(workflow.WorkflowSpecification):
    tasks = {
        ODE_FIT_JOBS.ode_fit: ODEFitTaskSpecification,
        ODE_FIT_JOBS.synthesis_spline: SynthesisSplineTaskSpecification,
    }


@dataclass
class ODEFitData:
    rates_version: str = field(default='best')
    waning_version: str = field(default='best')
    location_set_version_id: int = field(default=0)
    location_set_file: str = field(default='')
    output_root: str = field(default='')
    output_format: str = field(default='csv')
    n_draws: int = field(default=100)
    run_counties: bool = field(init=False)
    drop_locations: list = field(default_factory=list)

    def __post_init__(self):
        self.run_counties = self.location_set_version_id in [841, 920]

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        return {k: v for k, v in utilities.asdict(self).items() if k != 'run_counties'}


Sampleable = Union[Tuple[float, float], float]


@dataclass
class ODEFitParameters:
    alpha: Sampleable = field(default=(0.9, 1.0))
    sigma: Sampleable = field(default=(0.2, 1/3))
    gamma: Sampleable = field(default=(0.2, 1/3))
    pi: Sampleable = field(default=(0.01, 0.1))

    kappa_none: Sampleable = field(default=0.0)
    kappa_ancestral: Sampleable = field(default=1.0)
    kappa_alpha: Sampleable = field(default=1.0)
    kappa_beta: Sampleable = field(default=1.0)
    kappa_gamma: Sampleable = field(default=1.0)
    kappa_delta: Sampleable = field(default=1.0)
    kappa_other: Sampleable = field(default=1.0)
    kappa_omega: Sampleable = field(default=1.0)

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        return utilities.asdict(self)


class ODEFitSpecification(utilities.Specification):

    def __init__(self,
                 data: ODEFitData,
                 workflow: ODEFitWorkflowSpecification,
                 parameters: ODEFitParameters):
        self._data = data
        self._workflow = workflow
        self._parameters = parameters

    @classmethod
    def parse_spec_dict(cls, specification_dict: Dict) -> Tuple:
        sub_specs = {
            'data': ODEFitData,
            'workflow': ODEFitWorkflowSpecification,
            'regression_parameters': ODEFitParameters,
        }
        for key, spec_class in list(sub_specs.items()):  # We're dynamically altering. Copy with list
            spec_dict = utilities.filter_to_spec_fields(
                specification_dict.get(key, {}),
                spec_class(),
            )
            sub_specs[key] = spec_class(**spec_dict)

        return tuple(sub_specs.values())

    @property
    def data(self) -> ODEFitData:
        return self._data

    @property
    def workflow(self) -> ODEFitWorkflowSpecification:
        return self._workflow

    @property
    def parameters(self) -> ODEFitParameters:
        return self._parameters

    def to_dict(self) -> Dict:
        spec = {
            'data': self.data.to_dict(),
            'workflow': self.workflow.to_dict(),
            'regression_parameters': self.parameters.to_dict(),

        }
        return spec
