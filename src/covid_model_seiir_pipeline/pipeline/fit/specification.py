from dataclasses import dataclass, field
from typing import Dict, NamedTuple, Tuple, Union

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
    seir_preprocess_version: str = field(default='best')
    output_root: str = field(default='')
    output_format: str = field(default='csv')
    n_draws: int = field(default=100)

    def to_dict(self) -> Dict:
        return utilities.asdict(self)


Sampleable = Union[Tuple[float, float], float]


@dataclass
class RatesParameters:

    dummy: Sampleable = field(default=(0.0, 1.0))

    def to_dict(self) -> Dict:
        return utilities.asdict(self)


@dataclass
class FitParameters:
    alpha: Sampleable = field(default=(0.9, 1.0))
    sigma: Sampleable = field(default=(0.2, 1/3))
    gamma: Sampleable = field(default=(0.2, 1/3))
    pi: Sampleable = field(default=(0.01, 0.1))

    kappa_none_infection: Sampleable = field(default=0.0)
    kappa_ancestral_infection: Sampleable = field(default=1.0)
    kappa_alpha_infection: Sampleable = field(default=1.0)
    kappa_beta_infection: Sampleable = field(default=1.0)
    kappa_gamma_infection: Sampleable = field(default=1.0)
    kappa_delta_infection: Sampleable = field(default=1.0)
    kappa_other_infection: Sampleable = field(default=1.0)
    kappa_omega_infection: Sampleable = field(default=1.0)

    kappa_none_death: Sampleable = field(default=0.0)
    kappa_ancestral_death: Sampleable = field(default=1.0)
    kappa_alpha_death: Sampleable = field(default=1.0)
    kappa_beta_death: Sampleable = field(default=1.0)
    kappa_gamma_death: Sampleable = field(default=1.0)
    kappa_delta_death: Sampleable = field(default=1.0)
    kappa_other_death: Sampleable = field(default=1.0)
    kappa_omega_death: Sampleable = field(default=1.0)

    kappa_none_admission: Sampleable = field(default=0.0)
    kappa_ancestral_admission: Sampleable = field(default=1.0)
    kappa_alpha_admission: Sampleable = field(default=1.0)
    kappa_beta_admission: Sampleable = field(default=1.0)
    kappa_gamma_admission: Sampleable = field(default=1.0)
    kappa_delta_admission: Sampleable = field(default=1.0)
    kappa_other_admission: Sampleable = field(default=1.0)
    kappa_omega_admission: Sampleable = field(default=1.0)

    kappa_none_case: Sampleable = field(default=0.0)
    kappa_ancestral_case: Sampleable = field(default=1.0)
    kappa_alpha_case: Sampleable = field(default=1.0)
    kappa_beta_case: Sampleable = field(default=1.0)
    kappa_gamma_case: Sampleable = field(default=1.0)
    kappa_delta_case: Sampleable = field(default=1.0)
    kappa_other_case: Sampleable = field(default=1.0)
    kappa_omega_case: Sampleable = field(default=1.0)

    def to_dict(self) -> Dict:
        return utilities.asdict(self)


class FitSpecification(utilities.Specification):

    def __init__(self,
                 data: FitData,
                 workflow: FitWorkflowSpecification,
                 rates_parameters: RatesParameters,
                 fit_parameters: FitParameters):
        self._data = data
        self._workflow = workflow
        self._rates_parameters = rates_parameters
        self._fit_parameters = fit_parameters

    @classmethod
    def parse_spec_dict(cls, spec_dict: Dict) -> Tuple:
        sub_specs = {
            'data': FitData,
            'workflow': FitWorkflowSpecification,
            'rates_parameters': RatesParameters,
            'fit_parameters': FitParameters,
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
    def rates_parameters(self) -> RatesParameters:
        return self._rates_parameters

    @property
    def fit_parameters(self) -> FitParameters:
        return self._fit_parameters

    def to_dict(self) -> Dict:
        spec = {
            'data': self.data.to_dict(),
            'workflow': self.workflow.to_dict(),
            'rates_parameters': self.rates_parameters.to_dict(),
            'fit_parameters': self.fit_parameters.to_dict(),
        }
        return spec
