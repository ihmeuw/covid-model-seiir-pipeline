from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Tuple, Union

from covid_shared import workflow
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    utilities,
)


# Specification type for uniform distributions
UniformSampleable = Union[Tuple[float, float], float]
# Specification type for choosing an int in a range
DiscreteUniformSampleable = Union[Tuple[int, int], int]
# Specification type for variant RRs
RRSampleable = Union[Tuple[float, float, float], float, str]


class __FitJobs(NamedTuple):
    covariate_pool: str
    beta_fit: str
    beta_fit_join_sentinel: str
    beta_fit_postprocess: str
    beta_fit_diagnostics: str


FIT_JOBS = __FitJobs(*__FitJobs._fields)


class CovariatePoolTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 5000
    default_m_mem_free = '30G'
    default_num_cores = 5


class BetaFitTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 5000
    default_m_mem_free = '50G'
    default_num_cores = 11


class JoinSentinelTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 100
    default_m_mem_free = '1G'
    default_num_cores = 1


class BetaFitPostprocessingTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 5000
    default_m_mem_free = '50G'
    default_num_cores = 11


class BetaFitDiagnosticsTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 5000
    default_m_mem_free = '50G'
    default_num_cores = 26


class FitWorkflowSpecification(workflow.WorkflowSpecification):
    tasks = {
        FIT_JOBS.covariate_pool: CovariatePoolTaskSpecification,
        FIT_JOBS.beta_fit: BetaFitTaskSpecification,
        FIT_JOBS.beta_fit_join_sentinel: JoinSentinelTaskSpecification,
        FIT_JOBS.beta_fit_postprocess: BetaFitPostprocessingTaskSpecification,
        FIT_JOBS.beta_fit_diagnostics: BetaFitDiagnosticsTaskSpecification,
    }


@dataclass
class FitData:
    seir_preprocess_version: str = field(default='best')
    output_root: str = field(default='')
    output_format: str = field(default='csv')
    n_draws: int = field(default=100)
    compare_version: str = field(default='')

    def to_dict(self) -> Dict:
        return utilities.asdict(self)


@dataclass
class RatesParameters:
    day_0: Union[str, pd.Timestamp] = field(default='2020-03-15')
    pred_start_date: Union[str, pd.Timestamp] = field(default='2019-11-01')
    pred_end_date: Union[str, pd.Timestamp] = field(default='2022-03-15')
    mortality_scalar: str = field(default='total')

    death_rate_threshold: float = field(default=1)
    variant_prevalence_threshold: float = field(default=0.1)
    inclusion_days: int = field(default=180)
    naive_ifr: float = field(default=0.005)

    exposure_to_admission: DiscreteUniformSampleable = field(default=(10, 14))
    exposure_to_seroconversion: DiscreteUniformSampleable = field(default=(14, 18))
    admission_to_death: DiscreteUniformSampleable = field(default=(12, 16))

    ifr_risk_ratio: RRSampleable = field(default='BMJ')
    ihr_risk_ratio: RRSampleable = field(default='BMJ')
    idr_risk_ratio: RRSampleable = field(default=1.0)

    omicron_ifr_scalar: UniformSampleable = field(default=1.0)
    omicron_ihr_scalar: UniformSampleable = field(default=1.0)
    omicron_idr_scalar: UniformSampleable = field(default=1.0)

    omega_like_omicron: bool = field(default=False)

    p_asymptomatic_pre_omicron: UniformSampleable = field(default=0.5)
    p_asymptomatic_post_omicron: UniformSampleable = field(default=0.9)
    minimum_asymptomatic_idr_fraction: UniformSampleable = field(default=0.1)
    maximum_asymptomatic_idr: UniformSampleable = field(default=0.2)

    day_inflection_options: List[str] = field(default_factory=list)

    def __post_init__(self):
        RISK_RATIOS = {
            'BMJ': {'mean': 1.64, 'lower': 1.32, 'upper': 2.04},  # https://www.bmj.com/content/372/bmj.n579
            'LSHTM': {'mean': 1.35, 'lower': 1.08, 'upper': 1.65},
            'Imperial': {'mean': 1.29, 'lower': 1.07, 'upper': 1.54},
            'Exeter': {'mean': 1.91, 'lower': 1.35, 'upper': 2.71},
            'PHE': {'mean': 1.65, 'lower': 1.21, 'upper': 2.25},
        }

        for ratio in ['ifr', 'ihr']:
            rr = getattr(self, f'{ratio}_risk_ratio')
            if isinstance(rr, str):
                rr = tuple(RISK_RATIOS[rr].values())
            setattr(self, f'{ratio}_risk_ratio', rr)

        if not self.day_inflection_options:
            self.day_inflection_options = [
                '2020-07-01', '2020-08-01', '2020-09-01',
                '2020-10-01', '2020-11-01', '2020-12-01',
                '2021-01-01', '2021-02-01', '2021-03-01',
            ]

        self.day_0: pd.Timestamp = pd.Timestamp(self.day_0)
        self.pred_start_date: pd.Timestamp = pd.Timestamp(self.pred_start_date)
        self.pred_end_date: pd.Timestamp = pd.Timestamp(self.pred_end_date)

    def to_dict(self) -> Dict:
        d = utilities.asdict(self)
        for element in ['day_0', 'pred_start_date', 'pred_end_date']:
            d[element] = d[element].strftime('%Y-%m-%d')
        return d


@dataclass
class FitParameters:
    omega_invasion_date: str = field(default='')
    alpha_all: UniformSampleable = field(default=(0.9, 1.0))

    pi_none: UniformSampleable = field(default=(0.01, 0.10))
    pi_ancestral: UniformSampleable = field(default=(0.01, 0.10))
    pi_alpha: UniformSampleable = field(default=(0.01, 0.10))
    pi_beta: UniformSampleable = field(default=(0.01, 0.10))
    pi_gamma: UniformSampleable = field(default=(0.01, 0.10))
    pi_delta: UniformSampleable = field(default=(0.01, 0.10))
    pi_omicron: UniformSampleable = field(default=(0.01, 0.10))
    pi_other: UniformSampleable = field(default=(0.01, 0.10))
    pi_omega: UniformSampleable = field(default=(0.01, 0.10))

    sigma_none: UniformSampleable = field(default=(0.2, 1 / 3))
    sigma_ancestral: UniformSampleable = field(default=(0.2, 1 / 3))
    sigma_alpha: UniformSampleable = field(default=(0.2, 1 / 3))
    sigma_beta: UniformSampleable = field(default=(0.2, 1 / 3))
    sigma_gamma: UniformSampleable = field(default=(0.2, 1 / 3))
    sigma_delta: UniformSampleable = field(default=(0.2, 1 / 3))
    sigma_omicron: UniformSampleable = field(default=(0.2, 1 / 3))
    sigma_other: UniformSampleable = field(default=(0.2, 1 / 3))
    sigma_omega: UniformSampleable = field(default=(0.2, 1 / 3))

    gamma_none: UniformSampleable = field(default=(0.2, 1 / 3))
    gamma_ancestral: UniformSampleable = field(default=(0.2, 1 / 3))
    gamma_alpha: UniformSampleable = field(default=(0.2, 1 / 3))
    gamma_beta: UniformSampleable = field(default=(0.2, 1 / 3))
    gamma_gamma: UniformSampleable = field(default=(0.2, 1 / 3))
    gamma_delta: UniformSampleable = field(default=(0.2, 1 / 3))
    gamma_omicron: UniformSampleable = field(default=(0.2, 1 / 3))
    gamma_other: UniformSampleable = field(default=(0.2, 1 / 3))
    gamma_omega: UniformSampleable = field(default=(0.2, 1 / 3))

    kappa_none: UniformSampleable = field(default=0.0)
    kappa_ancestral: UniformSampleable = field(default=1.0)
    kappa_alpha: UniformSampleable = field(default=1.0)
    kappa_beta: UniformSampleable = field(default=1.0)
    kappa_gamma: UniformSampleable = field(default=1.0)
    kappa_delta: UniformSampleable = field(default=1.0)
    kappa_omicron: UniformSampleable = field(default=1.0)
    kappa_other: UniformSampleable = field(default=1.0)
    kappa_omega: UniformSampleable = field(default=1.0)

    phi_none: UniformSampleable = field(default=0.0)
    phi_ancestral: UniformSampleable = field(default=1.0)
    phi_alpha: UniformSampleable = field(default=1.0)
    phi_beta: UniformSampleable = field(default=1.0)
    phi_gamma: UniformSampleable = field(default=1.0)
    phi_delta: UniformSampleable = field(default=1.0)
    phi_omicron: UniformSampleable = field(default=1.0)
    phi_other: UniformSampleable = field(default=1.0)
    phi_omega: UniformSampleable = field(default=1.0)

    def to_dict(self) -> Dict:
        return utilities.asdict(self)


@dataclass
class MeasureDownweights:
    death: List[Tuple[int, float]] = field(default_factory=list)
    admission: List[Tuple[int, float]] = field(default_factory=list)
    case: List[Tuple[int, float]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return utilities.asdict(self)


class FitSpecification(utilities.Specification):

    def __init__(self,
                 data: FitData,
                 workflow: FitWorkflowSpecification,
                 rates_parameters: RatesParameters,
                 fit_parameters: FitParameters,
                 measure_downweights: MeasureDownweights):
        self._data = data
        self._workflow = workflow
        self._rates_parameters = rates_parameters
        self._fit_parameters = fit_parameters
        self._measure_downweights = measure_downweights

    @classmethod
    def parse_spec_dict(cls, spec_dict: Dict) -> Tuple:
        sub_specs = {
            'data': FitData,
            'workflow': FitWorkflowSpecification,
            'rates_parameters': RatesParameters,
            'fit_parameters': FitParameters,
            'measure_downweights': MeasureDownweights,
        }
        for key, spec_class in list(sub_specs.items()):  # We're dynamically altering. Copy with list
            key_spec_dict = utilities.filter_to_spec_fields(
                spec_dict.get(key, {}),
                spec_class(),
            )
            sub_specs[key] = spec_class(**key_spec_dict)

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

    @property
    def measure_downweights(self) -> MeasureDownweights:
        return self._measure_downweights

    def to_dict(self) -> Dict:
        spec = {
            'data': self.data.to_dict(),
            'workflow': self.workflow.to_dict(),
            'rates_parameters': self.rates_parameters.to_dict(),
            'fit_parameters': self.fit_parameters.to_dict(),
            'measure_downweights': self.measure_downweights.to_dict(),
        }
        return spec
