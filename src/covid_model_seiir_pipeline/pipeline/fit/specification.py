from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Tuple, Union

from covid_shared import workflow

from covid_model_seiir_pipeline.lib import (
    utilities,
)


class __FitJobs(NamedTuple):
    beta_fit: str


FIT_JOBS = __FitJobs(*__FitJobs._fields)


class BetaFitTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 5000
    default_m_mem_free = '30G'
    default_num_cores = 5


class FitWorkflowSpecification(workflow.WorkflowSpecification):
    tasks = {
        FIT_JOBS.beta_fit: BetaFitTaskSpecification,
    }


@dataclass
class FitData:
    preprocessing_version: str = field(default='best')
    output_root: str = field(default='')
    output_format: str = field(default='csv')

    def to_dict(self) -> Dict:
        return utilities.asdict(self)


Sampleable = Union[Tuple[float, float], float]


@dataclass
class FitParameters:
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

    zeta_death_none: Sampleable = field(default=0.0)
    zeta_death_ancestral: Sampleable = field(default=1.0)
    zeta_death_alpha: Sampleable = field(default=1.0)
    zeta_death_beta: Sampleable = field(default=1.0)
    zeta_death_gamma: Sampleable = field(default=1.0)
    zeta_death_delta: Sampleable = field(default=1.0)
    zeta_death_other: Sampleable = field(default=1.0)
    zeta_death_omega: Sampleable = field(default=1.0)

    zeta_case_none: Sampleable = field(default=1.0)
    zeta_case_ancestral: Sampleable = field(default=1.0)
    zeta_case_alpha: Sampleable = field(default=1.0)
    zeta_case_beta: Sampleable = field(default=1.0)
    zeta_case_gamma: Sampleable = field(default=1.0)
    zeta_case_delta: Sampleable = field(default=1.0)
    zeta_case_other: Sampleable = field(default=1.0)
    zeta_case_omega: Sampleable = field(default=1.0)

    zeta_admission_none: Sampleable = field(default=1.0)
    zeta_admission_ancestral: Sampleable = field(default=1.0)
    zeta_admission_alpha: Sampleable = field(default=1.0)
    zeta_admission_beta: Sampleable = field(default=1.0)
    zeta_admission_gamma: Sampleable = field(default=1.0)
    zeta_admission_delta: Sampleable = field(default=1.0)
    zeta_admission_other: Sampleable = field(default=1.0)
    zeta_admission_omega: Sampleable = field(default=1.0)

    def to_dict(self) -> Dict:
        return utilities.asdict(self)


class FitSpecification(utilities.Specification):

    def __init__(self,
                 data: FitData,
                 workflow: FitWorkflowSpecification,
                 parameters: FitParameters):
        self._data = data
        self._workflow = workflow
        self._parameters = parameters

    @classmethod
    def parse_spec_dict(cls, spec_dict: Dict) -> Tuple:
        sub_specs = {
            'data': FitData,
            'workflow': FitWorkflowSpecification,
            'parameters': FitParameters,
        }
        for key, spec_class in list(sub_specs.items()):  # We're dynamically altering. Copy with list
            spec_dict = utilities.filter_to_spec_fields(
                spec_dict.get(key, {}),
                spec_class(),
            )
            sub_specs[key] = spec_class(**spec_dict)

        return tuple(sub_specs.values())

    @property
    def data(self) -> FitData:
        return self._data

    @property
    def workflow(self) -> FitWorkflowSpecification:
        return self._workflow

    @property
    def parameters(self) -> FitParameters:
        return self._parameters

    def to_dict(self) -> Dict:
        spec = {
            'data': self.data.to_dict(),
            'workflow': self.workflow.to_dict(),
            'parameters': self.parameters.to_dict(),
        }
        return spec
