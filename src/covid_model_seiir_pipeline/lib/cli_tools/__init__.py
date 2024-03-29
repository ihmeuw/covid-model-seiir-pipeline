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
    VersionInfo,

    with_specification,

    with_location_specification,
    with_version,

    with_task_preprocessing_version,
    with_task_fit_version,
    with_task_regression_version,
    with_task_forecast_version,
    with_task_postprocessing_version,
    with_task_diagnostics_version,

    with_task_counterfactual_version,
    with_task_oos_holdout_version,

    with_scenario,
    with_plot_type,
    with_measure,
    with_draw_id,
    with_name,
    with_progress_bar,
    add_preprocess_only,
)
from covid_model_seiir_pipeline.lib.cli_tools.utilities import (
    resolve_version_info,
    handle_exceptions,
    get_input_root,
    get_output_root,
    get_location_info,
)
from covid_model_seiir_pipeline.lib.cli_tools.performance_logger import (
    task_performance_logger,
)
