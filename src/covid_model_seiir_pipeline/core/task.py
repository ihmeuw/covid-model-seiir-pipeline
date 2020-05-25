from jobmon.client.swarm.workflow.bash_task import BashTask
from jobmon.client.swarm.executors.base import ExecutorParameters


ExecParams = ExecutorParameters(
    max_runtime_seconds=1000,
    j_resource=False,
    m_mem_free='20G',
    num_cores=3,
    queue='d.q'
)

ExecParamsPlotting = ExecutorParameters(
    max_runtime_seconds=int(60*60*5),
    j_resource=False,
    m_mem_free='20G',
    num_cores=3,
    queue='d.q'
)


class RegressionTask(BashTask):
    def __init__(self, draw_id, regression_version, **kwargs):

        self.draw_id = draw_id

        command = (
            "beta_regression " +
            f"--draw-id {self.draw_id} " +
            f"--regression-version {regression_version} "
        )

        super().__init__(
            command=command,
            name=f'seiir_regression_fit_{draw_id}',
            executor_parameters=ExecParams,
            max_attempts=1,
            **kwargs
        )


class ForecastTask(BashTask):
    def __init__(self, location_id, regression_version, forecast_version, **kwargs):

        self.location_id = location_id
        self.regression_version = regression_version
        self.forecast_version = forecast_version

        command = (
            "beta_forecast " +
            f"--location-id {self.location_id} " +
            f"--regression-version {regression_version} " +
            f"--forecast-version {forecast_version} "
        )
        super().__init__(
            command=command,
            name=f'seiir_forecast_location_{location_id}',
            executor_parameters=ExecParams,
            max_attempts=1,
            **kwargs
        )

    def add_splicer_task(self, add_diagnostic):
        tasks = []

        splicer_task = SplicerTask(
            location_id=self.location_id,
            regression_version=self.regression_version,
            forecast_version=self.forecast_version
        )
        self.add_downstream(splicer_task)
        if add_diagnostic:
            diagnostic_task = ForecastDiagnosticTask(
                location_id=self.location_id,
                regression_version=self.regression_version,
                forecast_version=self.forecast_version
            )
            splicer_task.add_downstream(diagnostic_task)
            tasks += [diagnostic_task]
        tasks += [splicer_task]
        return tasks


class SplicerTask(BashTask):
    def __init__(self, location_id, regression_version, forecast_version, **kwargs):

        self.location_id = location_id

        command = (
            "splice " +
            f"--location-id {self.location_id} " +
            f"--regression-version {regression_version} " +
            f"--forecast-version {forecast_version} "
        )

        super().__init__(
            command=command,
            name=f'seiir_splice_location_{location_id}',
            executor_parameters=ExecParams,
            max_attempts=1,
            **kwargs
        )


class RegressionDiagnosticTask(BashTask):
    def __init__(self, regression_version, **kwargs):

        self.regression_version = regression_version

        command = (
            "create_regression_diagnostics " +
            f"--regression-version {regression_version} "
        )

        super().__init__(
            command=command,
            name=f'seiir_regression_diagnostics',
            executor_parameters=ExecParamsPlotting,
            max_attempts=1,
            **kwargs
        )


class ScalingDiagnosticTask(BashTask):
    def __init__(self, regression_version, forecast_version, **kwargs):

        self.regression_version = regression_version
        self.forecast_version = forecast_version

        command = (
            "create_scaling_diagnostics " +
            f"--regression-version {regression_version} " +
            f"--forecast-version {forecast_version}"
        )

        super().__init__(
            command=command,
            name=f'seiir_scaling_diagnostics',
            executor_parameters=ExecParamsPlotting,
            max_attempts=1,
            **kwargs
        )


class ForecastDiagnosticTask(BashTask):
    def __init__(self, regression_version, forecast_version, location_id, **kwargs):

        self.regression_version = regression_version
        self.forecast_version = forecast_version
        self.location_id = location_id

        command = (
            "create_forecast_diagnostics " +
            f"--regression-version {regression_version} " +
            f"--forecast-version {forecast_version} " +
            f"--location-id {location_id} "
        )

        super().__init__(
            command=command,
            name=f'seiir_forecast_diagnostics',
            executor_parameters=ExecParamsPlotting,
            max_attempts=1,
            **kwargs
        )
