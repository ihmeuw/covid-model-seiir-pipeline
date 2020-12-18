


def do_postprocessing(app_metadata: cli_tools.Metadata,
                      postprocessing_specification: PostprocessingSpecification,
                      preprocess_only: bool):
    logger.debug('Starting postprocessing')
    forecast_specification = ForecastSpecification.from_path(
        Path(postprocessing_specification.data.forecast_version) / 'forecast_specification.yaml'
    )
    data_interface = ForecastDataInterface.from_specification(forecast_specification, postprocessing_specification)

    # Check scenario covariates the same as regression covariates and that
    # covariate data versions match.
    covariates = data_interface.check_covariates(forecast_specification.scenarios)

    data_interface.make_dirs()
    postprocessing_specification.dump(data_interface.postprocessing_paths.postprocessing_specification)

    if not preprocess_only:
        workflow = PostprocessingWorkflow(postprocessing_specification.data.output_root)
        workflow.attach_tasks(forecast_specification.scenarios, covariates)

        try:
            workflow.run()
        except WorkflowAlreadyComplete:
            logger.info('Workflow already complete')
