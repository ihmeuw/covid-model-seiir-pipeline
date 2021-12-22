from covid_model_seiir_pipeline.pipeline.forecasting.specification import (
    FORECAST_JOBS,
    ForecastSpecification,
)
from covid_model_seiir_pipeline.pipeline.forecasting.data import (
    ForecastDataInterface,
)
from covid_model_seiir_pipeline.pipeline.forecasting.task import (
    beta_residual_scaling,
    beta_forecast,
)
from covid_model_seiir_pipeline.pipeline.forecasting.main import (
    do_forecast,
    forecast,
)

SPECIFICATION = ForecastSpecification
COMMAND = forecast
APPLICATION_MAIN = do_forecast
TASKS = {
    FORECAST_JOBS.scaling: beta_residual_scaling,
    FORECAST_JOBS.forecast: beta_forecast,
}
