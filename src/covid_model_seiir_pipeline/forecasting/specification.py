from dataclasses import dataclass, field
import itertools
from typing import Dict, NamedTuple, Tuple, Union

from covid_model_seiir_pipeline.utilities import Specification, asdict
from covid_model_seiir_pipeline.workflow_tools.specification import TaskSpecification, WorkflowSpecification


class __ForecastJobs(NamedTuple):
    scaling: str = 'scaling'
    forecast: str = 'forecast'
    sentinel: str = 'sentinel'
    resample: str = 'resample'
    postprocess: str = 'postprocess'


FORECAST_JOBS = __ForecastJobs()


class ScalingTaskSpecification(TaskSpecification):
    """Specification of execution parameters for beta scaling tasks."""
    default_max_runtime_seconds = 5000
    default_m_mem_free = '5G'
    default_num_cores = 26


class ForecastTaskSpecification(TaskSpecification):
    """Specification of execution parameters for beta forecasting tasks."""
    default_max_runtime_seconds = 15000
    default_m_mem_free = '5G'
    default_num_cores = 1


class SentinelTaskSpecification(TaskSpecification):
    """Specification of execution parameters for workflow join tasks."""
    default_max_runtime_seconds = 1000
    default_m_mem_free = '1G'
    default_num_cores = 1


class ResampleTaskSpecification(TaskSpecification):
    """Specification of execution parameters for draw resample mapping tasks."""
    default_max_runtime_seconds = 5000
    default_m_mem_free = '50G'
    default_num_cores = 26


class PostprocessingTaskSpecification(TaskSpecification):
    """Specification of execution parameters for postprocessing tasks."""
    default_max_runtime_seconds = 15000
    default_m_mem_free = '150G'
    default_num_cores = 26


class ForecastWorkflowSpecification(WorkflowSpecification):
    """Specification of execution parameters for forecasting workflows."""

    tasks = {
        FORECAST_JOBS.scaling: ScalingTaskSpecification,
        FORECAST_JOBS.forecast: ForecastTaskSpecification,
        FORECAST_JOBS.sentinel: SentinelTaskSpecification,
        FORECAST_JOBS.resample: ResampleTaskSpecification,
        FORECAST_JOBS.postprocess: PostprocessingTaskSpecification,
    }


@dataclass
class ForecastData:
    """Specifies the inputs and outputs for a forecast."""
    regression_version: str = field(default='best')
    covariate_version: str = field(default='best')
    output_root: str = field(default='')

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        return asdict(self)


@dataclass
class ScenarioSpecification:
    """Forecasting specification for a scenario."""
    ALLOWED_ALGORITHMS = (
        'normal',
        'draw_level_mandate_reimposition',
    )
    ALLOWED_SOLVERS = (
        'RK45',
    )
    ALLOWED_SYSTEMS = (
        'normal',
        'vaccine',
    )
    ALLOWED_POPULATION_PARTITIONS = (
        'none',
        'high_and_low_risk',
    )
    BETA_SCALING_KEYS = (
        'window_size',
        'average_over_min',
        'average_over_max',
        'offset_deaths_lower',
        'offset_deaths_upper',
    )

    name: str = field(default='dummy_scenario')
    algorithm: str = field(default='normal')
    algorithm_params: Dict = field(default_factory=dict)
    solver: str = field(default='RK45')
    system: str = field(default='normal')
    system_params: Dict = field(default_factory=dict)
    population_partition: str = field(default='none')
    beta_scaling: Dict[str, int] = field(default_factory=dict)
    theta: Union[str, int] = field(default=0)
    covariates: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.algorithm not in self.ALLOWED_ALGORITHMS:
            raise ValueError(f'Unknown algorithm {self.algorithm} in scenario {self.name}. '
                             f'Allowed algorithms are {self.ALLOWED_ALGORITHMS}.')

        if self.solver not in self.ALLOWED_SOLVERS:
            raise ValueError(f'Unknown solver {self.solver} in scenario {self.name}. '
                             f'Allowed solvers are {self.ALLOWED_SOLVERS}.')

        if self.system not in self.ALLOWED_SYSTEMS:
            raise ValueError(f'Unknown system {self.system} in scenario {self.name}. '
                             f'Allowed solvers are {self.ALLOWED_SYSTEMS}.')

        if self.population_partition not in self.ALLOWED_POPULATION_PARTITIONS:
            raise ValueError(f'Unknown population partition {self.population_partition} in scenario {self.name}. '
                             f'Allowed partitions are {self.ALLOWED_POPULATION_PARTITIONS}.')

        bad_scaling_keys = set(self.beta_scaling).difference(self.BETA_SCALING_KEYS)
        if bad_scaling_keys:
            raise ValueError(f'Unknown beta scaling configuration option(s) {list(bad_scaling_keys)} '
                             f'in scenario {self.name}. Expected options: {self.BETA_SCALING_KEYS}.')

        window_size = self.beta_scaling.get('window_size', None)
        if window_size is None:
            self.beta_scaling['window_size'] = -1
        else:
            if not isinstance(window_size, int) or window_size < 0:
                raise TypeError(f'Beta scaling window must be a positive int indicating no scaling. '
                                f'Scaling window for scenario {self.name} is {window_size}.')

        average_min = self.beta_scaling.get('average_over_min', None)
        average_max = self.beta_scaling.get('average_over_max', None)
        if average_min is None and average_max is None:
            self.beta_scaling['average_over_min'] = 0
            self.beta_scaling['average_over_max'] = 0
        # TODO: more input validation.

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        return {k: v for k, v in asdict(self).items() if k != 'name'}


@dataclass
class PostprocessingSpecification:

    resampling: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class ForecastSpecification(Specification):
    """Specification for a beta forecast run."""

    def __init__(self, data: ForecastData,
                 workflow: ForecastWorkflowSpecification,
                 postprocessing: PostprocessingSpecification,
                 *scenarios: ScenarioSpecification):
        self._data = data
        self._workflow = workflow
        self._postprocessing = postprocessing

        self._scenarios = {s.name: s for s in scenarios}

    @classmethod
    def parse_spec_dict(cls, forecast_spec_dict: Dict) -> Tuple:
        """Construct forecast specification args from a dict."""
        data = ForecastData(**forecast_spec_dict.get('data', {}))
        workflow = ForecastWorkflowSpecification(**forecast_spec_dict.get('workflow', {}))
        postprocessing = PostprocessingSpecification(**forecast_spec_dict.get('postprocessing', {}))
        scenario_dicts = forecast_spec_dict.get('scenarios', {})
        scenarios = []
        for name, scenario_spec in scenario_dicts.items():
            scenarios.append(ScenarioSpecification(name, **scenario_spec))
        if not scenarios:
            scenarios.append(ScenarioSpecification())
        return tuple(itertools.chain([data, workflow, postprocessing], scenarios))

    @property
    def data(self) -> ForecastData:
        """The forecast data specification."""
        return self._data

    @property
    def workflow(self) -> ForecastWorkflowSpecification:
        """The workflow specification for the forecast."""
        return self._workflow

    @property
    def postprocessing(self) -> PostprocessingSpecification:
        return self._postprocessing

    @property
    def scenarios(self) -> Dict[str, ScenarioSpecification]:
        """The specification of all scenarios in the forecast."""
        return self._scenarios

    def to_dict(self):
        """Convert the specification to a dict."""
        return {
            'data': self.data.to_dict(),
            'workflow': self.workflow.to_dict(),
            'postprocessing': self.postprocessing.to_dict(),
            'scenarios': {k: v.to_dict() for k, v in self._scenarios.items()}
        }
