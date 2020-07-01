from pathlib import Path

import pandas

from covid_model_seiir_pipeline.marshall import (
    CSVMarshall,
    Keys as MKeys,
)
from covid_model_seiir_pipeline.paths import ODEPaths
from covid_model_seiir_pipeline.regression.data import RegressionDataInterface


class TestRegressionDataInterfaceIO:
    def test_ode_fit_io(self, tmpdir, tmpdir_file_count, beta_result):
        """
        Test I/O relating to the fit stage.

        This only includes loading files.
        """
        ode_paths = ODEPaths(Path(tmpdir))
        di = RegressionDataInterface(
            regression_paths=None,
            ode_paths=ode_paths,
            covariate_paths=None,
            ode_marshall=CSVMarshall(ode_paths.root_dir),
        )

        # Step 1: create files
        m = CSVMarshall(di.ode_paths.root_dir)
        m.dump(beta_result, key=MKeys.beta(draw_id=4))

        # Step 2: load files
        # Note: 10 corresponds to a value in the ode_fit_beta fixture
        loaded = di.load_ode_fits(draw_id=4, location_ids=[10])

        # Step 3: test
        # Note: load_ode_fits explicitly converts the "date" column
        assert pandas.core.dtypes.common.is_datetime64_any_dtype(loaded['date'])
        loaded['date'] = loaded['date'].dt.strftime("%Y-%m-%d")
        pandas.testing.assert_frame_equal(beta_result, loaded)
