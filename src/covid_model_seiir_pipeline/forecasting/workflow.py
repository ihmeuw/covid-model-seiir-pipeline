from typing import Dict, List

from jobmon.client import Workflow, BashTask
from jobmon.client.swarm.executors.base import ExecutorParameters
from jobmon.client.swarm.workflow.task_dag import DagExecutionStatus

from covid_model_seiir_pipeline import utilities
from covid_model_seiir_pipeline.forecasting import ForecastSpecification


class BetaForecastTaskTemplate:

    def __init__(self, task_registry: Dict[str, BashTask]):
        self.task_registry = task_registry
        self.command_template = (
            "beta_forecast " +
            "--location-id {location_id} " +
            "--forecast-version {forecast_version} " +
            "--scenario {scenario}"
        )
        self.params = ExecutorParameters(
            max_runtime_seconds=1000,
            j_resource=False,
            m_mem_free='20G',
            num_cores=3,
            queue='d.q'
        )

    @staticmethod
    def get_task_name(location_id: int, scenario: str) -> str:
        return f"forecast_{location_id}_{scenario}"

    def get_task(self, location_id: int, forecast_version: str, scenario: str):
        task_name = self.get_task_name(location_id, scenario)
        task = BashTask(
            command=self.command_template.format(location_id=location_id,
                                                 forecast_version=forecast_version,
                                                 scenario=scenario),
            name=task_name,
            executor_parameters=self.params,
            max_attempts=1
        )
        self.task_registry[task_name] = task
        return task


class SpliceTaskTemplate:

    def __init__(self, task_registry: Dict[str, BashTask]):
        self.task_registry = task_registry
        self.command_template = (
            "splice " +
            "--location-id {location_id} " +
            "--forecast-version {forecast_version} " +
            "--scenario {scenario}"
        )
        self.params = ExecutorParameters(
            max_runtime_seconds=1000,
            j_resource=False,
            m_mem_free='20G',
            num_cores=3,
            queue='d.q'
        )

    @staticmethod
    def get_task_name(location_id: int, scenario: str) -> str:
        return f"splice_{location_id}_{scenario}"

    def get_task(self, location_id: int, forecast_version: str, scenario: str):
        # get upstreams first
        upstream_name = BetaForecastTaskTemplate.get_task_name(location_id, scenario)
        upstream_task = self.task_registry[upstream_name]

        # now create task
        task_name = self.get_task_name(location_id, scenario)
        task = BashTask(
            command=self.command_template.format(location_id=location_id,
                                                 forecast_version=forecast_version,
                                                 scenario=scenario),
            name=task_name,
            executor_parameters=self.params,
            max_attempts=1,
            upstream_tasks=[upstream_task]
        )

        self.task_registry[task_name] = task
        return task


class ForecastWorkflow:

    def __init__(self, forecast_specification: ForecastSpecification, location_ids: List[int]
                 ) -> None:
        self.forecast_specification = forecast_specification
        self.location_ids = location_ids

        # if we need to build dependencies then the task registry must be shared between
        # task factories
        self._task_registry: Dict[str, BashTask] = {}
        self._beta_forecast_task_template = BetaForecastTaskTemplate(self._task_registry)
        self._splice_task_template = SpliceTaskTemplate(self._task_registry)

        workflow_args = f'seiir-forecast-{forecast_specification.data.output_root}'
        stdout, stderr = utilities.make_log_dirs(
            self.forecast_specification.data.output_root, prefix='jobmon'
        )

        self.workflow = Workflow(
            workflow_args=workflow_args,
            project="proj_covid",
            stderr=stderr,
            stdout=stdout,
            seconds_until_timeout=60*60*24,
            resume=True
        )

    def attach_scenario_tasks(self):
        for location_id in self.location_ids:
            for scenario in self.forecast_specification.scenarios.keys():
                task = self._beta_forecast_task_template.get_task(
                    location_id=location_id,
                    forecast_version=self.forecast_specification.data.output_root,
                    scenario=scenario
                )
                self.workflow.add_task(task)

    def attach_splice_tasks(self):
        for location_id in self.location_ids:
            for scenario in self.forecast_specification.scenarios.keys():
                task = self._splice_task_template.get_task(
                    location_id=location_id,
                    forecast_version=self.forecast_specification.data.output_root,
                    scenario=scenario
                )
                self.workflow.add_task(task)

    def run(self):
        execution_status = self.workflow.run()
        if execution_status != DagExecutionStatus.SUCCEEDED:
            raise RuntimeError("Workflow failed. Check database or logs for errors")
