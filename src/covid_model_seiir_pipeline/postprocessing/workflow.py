from typing import Dict

from jobmon.client import BashTask
from jobmon.client.swarm.executors.base import ExecutorParameters


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
            m_mem_free='20G',
            num_cores=3,
            queue='d.q'
        )

    @staticmethod
    def get_task_name(location_id: int, scenario: str) -> str:
        return f"splice_{scenario}_{location_id}"

    def get_task(self, location_id: int, forecast_version: str, scenario: str):
        # get upstreams first

        # now create task
        task_name = self.get_task_name(location_id, scenario)
        task = BashTask(
            command=self.command_template.format(location_id=location_id,
                                                 forecast_version=forecast_version,
                                                 scenario=scenario),
            name=task_name,
            executor_parameters=self.params,
            max_attempts=1,
        )

        self.task_registry[task_name] = task
        return task
