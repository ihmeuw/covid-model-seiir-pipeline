from pathlib import Path

from covid_shared import paths

from covid_model_seiir_pipeline.lib.cli_tools.utilities import (
    get_input_root
)


def test_get_input_root():
    p = get_input_root(None, None, paths.INFECTIONATOR_OUTPUTS)
    assert p == (paths.INFECTIONATOR_OUTPUTS / paths.BEST_LINK).resolve()

    p = get_input_root(None, 'my_test_root', paths.INFECTIONATOR_OUTPUTS)
    assert p == (paths.INFECTIONATOR_OUTPUTS / 'my_test_root').resolve()

    p = get_input_root('my_cli_test_root', None, paths.INFECTIONATOR_OUTPUTS)
    assert p == (paths.INFECTIONATOR_OUTPUTS / 'my_cli_test_root').resolve()

    p = get_input_root('my_cli_test_root', 'my_test_root', paths.INFECTIONATOR_OUTPUTS)
    assert p == (paths.INFECTIONATOR_OUTPUTS / 'my_cli_test_root').resolve()

    p = get_input_root(None, '/my/full/test/root', paths.INFECTIONATOR_OUTPUTS)
    assert p == Path('/my/full/test/root')

    p = get_input_root('/my/full/cli/test/root', None, paths.INFECTIONATOR_OUTPUTS)
    assert p == Path('/my/full/cli/test/root')

    p = get_input_root('/my/full/cli/test/root', '/my/full/test/root', paths.INFECTIONATOR_OUTPUTS)
    assert p == Path('/my/full/cli/test/root')
