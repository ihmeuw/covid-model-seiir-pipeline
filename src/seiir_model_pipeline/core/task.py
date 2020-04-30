from jobmon.client.swarm.workflow.bash_task import BashTask
from jobmon.client.swarm.executors.base import ExecutorParameters


ExecParams = ExecutorParameters(
    max_runtime_seconds=1000,
    j_resource=False,
    m_mem_free='20G',
    num_cores=3
)


class DrawTask(BashTask):
    """
    A draw task, subclass of jobmon BashTask
    """
    def __init__(self, draw_id, directories, output_version, warm_start):

        self.draw_id = draw_id

        command = (
            "run_one_draw " +
            f"--draw-id {self.draw_id} " +
            f"--output-version {output_version} "
        )
        if warm_start:
            command += "--warm-start"

        super().__init__(
            command=command,
            name=f'seiir_model_run_{directories.output_version}_{draw_id}',
            executor_parameters=ExecParams
        )
