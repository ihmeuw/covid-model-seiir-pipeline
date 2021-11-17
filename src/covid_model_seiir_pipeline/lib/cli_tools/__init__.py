# This is just exposing the api from this namespace.
from covid_shared.cli_tools import (
    pass_run_metadata,
    add_verbose_and_with_debugger,
    with_mark_best,
    with_production_tag,
    add_output_options,
    setup_directory_structure,
    make_run_directory,
    configure_logging_to_terminal,
    configure_logging_to_files,
    monitor_application,
    finish_application,
    Metadata,
    RunMetadata,
)
from covid_model_seiir_pipeline.lib.cli_tools.decorators import (
    with_fit_specification,
    with_regression_specification,
    with_forecast_specification,
    with_postprocessing_specification,
    with_diagnostics_specification,
    with_predictive_validity_specification,

    with_infection_version,
    with_covariates_version,
    with_waning_version,
    with_variant_version,
    with_mortality_ratio_version,
    with_priors_version,
    with_coefficient_version,
    with_location_specification,
    with_regression_version,
    with_forecast_version,

    with_task_preprocessing_version,
    with_task_fit_version,
    with_task_regression_version,
    with_task_forecast_version,
    with_task_postprocessing_version,
    with_task_diagnostics_version,

    with_scenario,
    with_measure,
    with_draw_id,
    with_name,
    with_progress_bar,
    add_preprocess_only,
)
from covid_model_seiir_pipeline.lib.cli_tools.utilities import (
    VersionInfo,
    resolve_version_info,
    handle_exceptions,
    get_input_root,
    get_output_root,
    get_location_info,
)
from covid_model_seiir_pipeline.lib.cli_tools.performance_logger import (
    task_performance_logger,
)
