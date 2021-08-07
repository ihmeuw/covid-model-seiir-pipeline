from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

from covid_shared import workflow

from covid_model_seiir_pipeline.lib import (
    utilities,
)


class __DiagnosticsJobs(NamedTuple):
    grid_plots: str = 'grid_plots'
    compare_csv: str = 'cumulative_deaths_compare_csv'


DIAGNOSTICS_JOBS = __DiagnosticsJobs()


class GridPlotsTaskSpecification(workflow.TaskSpecification):
    """Specification of execution parameters for grid plots tasks."""
    default_max_runtime_seconds = 5000
    default_m_mem_free = '200G'
    default_num_cores = 70


class CumulativeDeathsCompareCSVTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 500
    default_m_mem_free = '10G'
    default_num_cores = 1


class DiagnosticsWorkflowSpecification(workflow.WorkflowSpecification):
    """Specification of execution parameters for diagnostics workflows."""

    tasks = {
        DIAGNOSTICS_JOBS.grid_plots: GridPlotsTaskSpecification,
        DIAGNOSTICS_JOBS.compare_csv: CumulativeDeathsCompareCSVTaskSpecification,
    }


@dataclass
class DiagnosticsData:
    """Specifies the inputs and outputs for postprocessing."""
    output_root: str = field(default='')

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        return utilities.asdict(self)


class ComparatorSpecification:

    def __init__(self,
                 version: str = 'best',
                 scenarios: Dict[str, str] = None):
        if scenarios is None:
            raise
        self.version = version
        self.scenarios = scenarios

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


class GridPlotsSpecification:

    def __init__(self,
                 name: str = '',
                 date_start: str = '2020-03-01',
                 date_end: str = '2020-12-31',
                 comparators: List[Dict] = None):
        if comparators is None:
            raise
        self.name = name
        self.date_start = date_start
        self.date_end = date_end
        self.comparators = [ComparatorSpecification(**comparator) for comparator in comparators]

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        output_dict = {k: v for k, v in self.__dict__.items() if k != 'comparators'}
        output_dict['comparators'] = [comparator.to_dict() for comparator in self.comparators]
        return output_dict


class CumulativeDeathsCompareCSVSpecification:

    def __init__(self,
                 dates: List[str] = None,
                 comparators: List[Dict] = None):
        if dates is None or comparators is None:
            raise
        self.dates = dates
        self.comparators = [ComparatorSpecification(**comparator) for comparator in comparators]

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        output_dict = {k: v for k, v in self.__dict__.items() if k != 'comparators'}
        output_dict['comparators'] = [comparator.to_dict() for comparator in self.comparators]
        return output_dict


class DiagnosticsSpecification(utilities.Specification):

    def __init__(self,
                 data: DiagnosticsData,
                 workflow: DiagnosticsWorkflowSpecification,
                 cumulative_deaths_compare_csv: Optional[CumulativeDeathsCompareCSVSpecification],
                 grid_plots: List[GridPlotsSpecification]):
        self._data = data
        self._workflow = workflow
        self._cumulative_deaths_compare_csv = cumulative_deaths_compare_csv
        self._grid_plots = grid_plots

    @classmethod
    def parse_spec_dict(cls, specification: Dict) -> Tuple:
        data = DiagnosticsData(**specification.get('data', {}))
        workflow = DiagnosticsWorkflowSpecification(**specification.get('workflow', {}))
        compare_csv = CumulativeDeathsCompareCSVSpecification(**specification.get('cumulative_deaths_compare_csv', None))
        grid_plots_configs = specification.get('grid_plots', [])
        grid_plots = [GridPlotsSpecification(**grid_plots_config) for grid_plots_config in grid_plots_configs]
        return data, workflow, compare_csv, grid_plots

    @property
    def data(self) -> DiagnosticsData:
        return self._data

    @property
    def workflow(self) -> DiagnosticsWorkflowSpecification:
        return self._workflow

    @property
    def cumulative_deaths_compare_csv(self) -> Union[CumulativeDeathsCompareCSVSpecification, None]:
        return self._cumulative_deaths_compare_csv

    @property
    def grid_plots(self) -> List[GridPlotsSpecification]:
        return self._grid_plots

    def to_dict(self):
        """Convert the specification to a dict."""
        out = {
            'data': self.data.to_dict(),
            'workflow': self.workflow.to_dict(),
            'grid_plots': [grid_plots_config.to_dict() for grid_plots_config in self.grid_plots],
        }
        if self.cumulative_deaths_compare_csv:
            out['cumulative_deaths_compare_csv'] = self.cumulative_deaths_compare_csv.to_dict()
        return out
