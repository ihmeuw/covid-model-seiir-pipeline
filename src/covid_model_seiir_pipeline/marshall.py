from contextlib import contextmanager
import io
from pathlib import Path
import zipfile

import pandas

from covid_model_seiir_pipeline.paths import (
    Paths,
    DRAW_FILE_TEMPLATE,
)
from covid_shared.shell_tools import mkdir


class DataTypes:
    """
    Enumerations of data types as understood by the Marshall interface.
    """
    coefficient = "coefficients"
    date = "dates"
    fit_beta = "betas"  # TODO: unique name after forecast is done
    parameter = "parameters"
    regression_beta = "betas"  # TODO: unique name after forecast is done

    # types which serialize to a DataFrame
    DataFrame_types = frozenset([
        fit_beta,
        parameter,
        date,
        coefficient,
        regression_beta
    ])

class Keys:
    def __init__(self, data_type, template, **key_args):
        self.data_type = data_type
        self.template = template
        self.key_args = key_args

    @classmethod
    def coefficient(cls, draw_id):
        return cls(DataTypes.coefficient, DRAW_FILE_TEMPLATE, draw_id=draw_id)

    @classmethod
    def date(cls, draw_id):
        return cls(DataTypes.date, DRAW_FILE_TEMPLATE, draw_id=draw_id)

    @classmethod
    def fit_beta(cls, draw_id):
        return cls(DataTypes.fit_beta, DRAW_FILE_TEMPLATE, draw_id=draw_id)

    @classmethod
    def parameter(cls, draw_id):
        return cls(DataTypes.parameter, DRAW_FILE_TEMPLATE, draw_id=draw_id)

    @classmethod
    def regression_beta(cls, draw_id):
        return cls(DataTypes.regression_beta, DRAW_FILE_TEMPLATE, draw_id=draw_id)

    @property
    def key(self):
        return self.template.format(**self.key_args)

    @property
    def seed(self):
        """
        Returns a seed value for enabling concurrency.

        This is a sort of hack - we embed the concurrency story into the Keys
        class which can then inform other things.
        """
        assert list(self.key_args) == ['draw_id'], 'TODO - expand Keys.seed'
        return "draw-{draw_id}".format(**self.key_args)

    def __repr__(self):
        return f"Keys({self.data_type!r}, {self.template!r}, **{self.key_args!r})"


class CSVMarshall:
    """
    Marshalls DataFrames to/from CSV files.

    This implementation directly mirrors existing behavior but does so within a
    new marshalling interface.
    """
    # interface methods
    def dump(self, data: pandas.DataFrame, key):
        path = self.resolve_key(key)
        if not path.parent.is_dir():
            mkdir(path.parent)
        else:
            if path.exists():
                msg = f"Cannot dump data for key {key} - would overwrite"
                raise LookupError(msg)

        data.to_csv(path, index=False)

    def load(self, key):
        path = self.resolve_key(key)
        return pandas.read_csv(path)

    def resolve_key(self, key):
        if key.data_type in DataTypes.DataFrame_types:
            path = (self.root / key.data_type / key.key).with_suffix(".csv")
        else:
            msg = f"Invalid 'type' of data: {key.data_type}"
            raise ValueError(msg)

        return path

    # non-interface methods
    @classmethod
    def from_paths(cls, paths: Paths):
        return cls(paths.root_dir)

    def __init__(self, root: Path):
        self.root = root


class ZipMarshall:
    # interface methods
    def dump(self, data: pandas.DataFrame, key):
        seed, path = self.resolve_key(key)
        with zipfile.ZipFile(self.zip(seed), mode='a') as container:
            with self._open_no_overwrite(container, path, key) as outf:
                # stream writes through a wrapper that does str => bytes conversion
                # chunksize denotes number of rows to write at once. it is not
                # clear to me that 5 is at all a good number
                wrapper = io.TextIOWrapper(outf, write_through=True)
                data.to_csv(wrapper, index=False, chunksize=5)

    def load(self, key):
        seed, path = self.resolve_key(key)
        with zipfile.ZipFile(self.zip(seed), mode='r') as container:
            with container.open(path, 'r') as inf:
                return pandas.read_csv(inf)

    def resolve_key(self, key):
        if key.data_type in DataTypes.DataFrame_types:
            path = f"{key.data_type}/{key.key}.csv"
            seed = key.seed
        else:
            msg = f"Invalid 'type' of data: {key.data_type}"
            raise ValueError(msg)

        return seed, path

    # non-interface methods
    @classmethod
    def from_paths(cls, paths: Paths):
        # technically unsafe - {} values in the root_dir could break interpolation
        zip_path = str(paths.root_dir / "seiir-data-{{}}.zip")
        return cls(zip_path)

    def __init__(self, zip_template: str):
        self.zip_template = zip_template

    def zip(self, seed: str):
        return self.zip_template.format(seed)

    @contextmanager
    def _open_no_overwrite(self, zip_container, path, key):
        try:
            zip_container.getinfo(path)
        except KeyError:
            pass  # file does not exist - everything OK
        else:
            raise LookupError(f"Cannot dump data for key {key} - would overwrite")

        with zip_container.open(path, 'w') as outf:
            yield outf
