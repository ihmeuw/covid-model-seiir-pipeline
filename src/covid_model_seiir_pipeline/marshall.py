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
    beta = "betas"
    coefficient = "coefficients"
    parameter = "parameters"
    date = "dates"


class Keys:
    def __init__(self, data_type, template, **key_args):
        self.data_type = data_type
        self.template = template
        self.key_args = key_args

    @classmethod
    def parameter(cls, draw_id):
        return cls(DataTypes.parameter, DRAW_FILE_TEMPLATE, draw_id=draw_id)

    @classmethod
    def beta(cls, draw_id):
        return cls(DataTypes.beta, DRAW_FILE_TEMPLATE, draw_id=draw_id)

    @classmethod
    def coefficient(cls, draw_id):
        return cls(DataTypes.coefficient, DRAW_FILE_TEMPLATE, draw_id=draw_id)

    @classmethod
    def date(cls, draw_id):
        return cls(DataTypes.date, DRAW_FILE_TEMPLATE, draw_id=draw_id)

    @property
    def key(self):
        return self.template.format(**self.key_args)

    def __repr__(self):
        return f"Keys({self.data_type!r}, {self.template!r}, **{self.key_args!r})"


class CSVMarshall:
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
        if key.data_type in {DataTypes.beta, DataTypes.parameter,
                             DataTypes.date, DataTypes.coefficient}:
            path = (self.root / key.data_type / key.key).with_suffix(".csv")
        else:
            msg = f"Invalid 'type' of data: {key.data_type}"
            raise ValueError(msg)

        return path

    # non-interface methods
    @classmethod
    def from_paths(cls, paths: Paths):
        return cls(paths)

    def __init__(self, root: Paths):
        self.root = root
