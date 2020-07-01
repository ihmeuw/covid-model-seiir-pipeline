from pathlib import Path

from covid_shared import cli_tools

from covid_model_seiir_pipeline.paths import ForecastPaths
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.forecasting.workflow import ForecastWorkflow


def do_beta_forecast(app_metadata: cli_tools.Metadata,
                     forecast_specification: ForecastSpecification):

    return
    # for scenario in forecast_specification.scenarios.keys():
    #     scenario_root = output_root / scenario
    #
    #     # init high level objects
    #     forecast_paths = ForecastPaths(scenario_root, read_only=False)
    #     data_interface = ForecastDataInterface(
    #         forecast_root=scenario_root,
    #         regression_root=Path(regression_specification.data.output_root),
    #         ode_fit_root=Path(ode_fit_spec.data.output_root),
    #         infection_root=Path(ode_fit_spec.data.infection_version),
    #         location_file=Path(ode_fit_spec.data.location_set_file)
    #     )
    #
    #     # build directory structure
    #     location_ids = data_interface.load_location_ids()
    #     forecast_paths.make_dirs(location_ids)

    # build workflow and launch
    # forecast_wf = ForecastWorkflow(forecast_specification, location_ids)
    # forecast_wf.attach_scenario_tasks()
    # forecast_wf.run()
