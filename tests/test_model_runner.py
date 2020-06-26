from covid_model_seiir_pipeline.model_runner import ModelRunner


def test_instantiation():
    """
    Test ModelRunner instantiation.

    This indirectly serves as a sort of integration test as ModelRunner imports
    all the modules that were originally merged from covid_model_seiir.
    """
    mr = ModelRunner()
