from pathlib import Path

import pandas

from covid_model_seiir_pipeline.marshall import CSVMarshall
from covid_model_seiir_pipeline.ode_fit.data import ODEDataInterface
from covid_model_seiir_pipeline import paths


class TestODEDataInterfaceIO:
    def test_created_files(self, tmpdir, tmpdir_file_count):
        """
        Integration test for measuring files actually created by ODEDataInterface.
        """
        di = ODEDataInterface(ode_paths=paths.ODEPaths(tmpdir),
                              infection_paths=None,
                              ode_marshall=CSVMarshall(Path(tmpdir)),
                              )

        # Step 1: count files
        assert tmpdir_file_count() == 0, "Files somehow already exist in storage dir"

        # Step 2: generate data
        def get_draw_files(id):
            "Return a beta fit file, a beta param file, and a date file."
            # TODO: the save methods don't currently check any type of schema.
            # If they ever do these DataFrame's probably won't pass validation
            bff = pandas.DataFrame({'type': ['fit'], 'secret': [id]})
            bpf = pandas.DataFrame({'type': ['param'], 'secret': [id]})
            df = pandas.DataFrame({'type': ['date'], 'secret': [id]})
            return bff, bpf, df

        bff1, bpf1, df1 = get_draw_files(1)
        bff2, bpf2, df2 = get_draw_files(2)
        bff3, bpf3, df3 = get_draw_files(3)

        # Step 3: save files

        # step 3a: make the directories the files have to be in
        (tmpdir / "betas").mkdir()
        (tmpdir / "dates").mkdir()
        (tmpdir / "parameters").mkdir()

        # step 3b: save files with API
        di.save_beta_fit_file(bff1, draw_id=1)
        di.save_beta_fit_file(bff2, draw_id=2)
        di.save_beta_fit_file(bff2, draw_id=3)

        di.save_draw_beta_param_file(bpf1, draw_id=1)
        di.save_draw_beta_param_file(bpf2, draw_id=2)
        di.save_draw_beta_param_file(bpf2, draw_id=3)

        di.save_draw_date_file(df1, draw_id=1)
        di.save_draw_date_file(df2, draw_id=2)
        di.save_draw_date_file(df3, draw_id=3)

        # Step 4: count files (again)
        assert tmpdir_file_count() == 9
