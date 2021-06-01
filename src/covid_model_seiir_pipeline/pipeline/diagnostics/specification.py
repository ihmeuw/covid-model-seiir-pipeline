from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Tuple

from covid_shared import workflow

from covid_model_seiir_pipeline.lib import (
    utilities,
)


class __DiagnosticsJobs(NamedTuple):
    grid_plots: str = 'grid_plots'


DIAGNOSTICS_JOBS = __DiagnosticsJobs()


class GridPlotsTaskSpecification(workflow.TaskSpecification):
    """Specification of execution parameters for grid plots tasks."""
    default_max_runtime_seconds = 5000
    default_m_mem_free = '200G'
    default_num_cores = 70


class DiagnosticsWorkflowSpecification(workflow.WorkflowSpecification):
    """Specification of execution parameters for diagnostics workflows."""

    tasks = {
        DIAGNOSTICS_JOBS.grid_plots: GridPlotsTaskSpecification,
    }


@dataclass
class DiagnosticsData:
    """Specifies the inputs and outputs for postprocessing."""
    output_root: str = field(default='')

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        return utilities.asdict(self)


class GridPlotsComparatorSpecification:

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
        self.comparators = [GridPlotsComparatorSpecification(**comparator) for comparator in comparators]

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        output_dict = {k: v for k, v in self.__dict__.items() if k != 'comparators'}
        output_dict['comparators'] = [comparator.to_dict() for comparator in self.comparators]
        return output_dict


class DiagnosticsSpecification(utilities.Specification):

    def __init__(self,
                 data: DiagnosticsData,
                 workflow: DiagnosticsWorkflowSpecification,
                 grid_plots: List[GridPlotsSpecification]):
        self._data = data
        self._workflow = workflow
        self._grid_plots = grid_plots

    @classmethod
    def parse_spec_dict(cls, specification: Dict) -> Tuple:
        data = DiagnosticsData(**specification.get('data', {}))
        workflow = DiagnosticsWorkflowSpecification(**specification.get('workflow', {}))
        grid_plots_configs = specification.get('grid_plots', [])
        grid_plots = [GridPlotsSpecification(**grid_plots_config) for grid_plots_config in grid_plots_configs]
        return data, workflow, grid_plots

    @property
    def data(self) -> DiagnosticsData:
        return self._data

    @property
    def workflow(self) -> DiagnosticsWorkflowSpecification:
        return self._workflow

    @property
    def grid_plots(self) -> List[GridPlotsSpecification]:
        return self._grid_plots

    def to_dict(self):
        """Convert the specification to a dict."""
        return {
            'data': self.data.to_dict(),
            'workflow': self.workflow.to_dict(),
            'grid_plots': [grid_plots_config.to_dict() for grid_plots_config in self.grid_plots],
        }
