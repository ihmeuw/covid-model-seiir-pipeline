from pathlib import Path

from covid_shared import cli_tools

from covid_model_seiir_pipeline.paths import ForecastPaths
from covid_model_seiir_pipeline.forecasting import ForecastSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface


def do_beta_forecast(app_metadata: cli_tools.Metadata,
                     forecast_specification: ForecastSpecification,
                     output_root: Path):

    for scenario in forecast_specification.scenarios.keys():
        scenario_root = output_root / scenario

        # init high level objects
        forecast_paths = ForecastPaths(scenario_root, read_only=False)
        data_interface = ForecastDataInterface(forecast_specification.data)

        # build directory structure
        location_ids = data_interface.load_location_ids()
        forecast_paths.make_dirs(location_ids)
