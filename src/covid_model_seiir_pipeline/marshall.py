from collections import defaultdict
from contextlib import contextmanager
import io
from pathlib import Path
import zipfile

import h5py
import numpy
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


class Hdf5Marshall:
    """
    Marshalls data to/from HDF5 format.

    HDF is a typed dataset format that brings several additional steps to
    successful marshalling.

    For the uninitiated, HDF5

    * stores all data in a single file
    * supports multiple sets of data (Dataset)
    * supports grouping of data (groups) which are analogous to directories

    Compression is available, applied at the Dataset level.

    http://docs.h5py.org/en/stable/

    > groups work like directories and datasets work like numpy arrays
    http://docs.h5py.org/en/stable/quick.html#core-concepts
    """
    # special names for Datasets of metadata used to help marshalling process
    file_version_attr = 'file_version'
    file_version_id = 1
    load_order_ds_name = "load_order"
    column_names_ds_attr = "column_names"
    column_names_order = "returned_column_names"
    # all datasets are named after their dtype
    # attributes are used to describe the column names for self-descriptiveness
    dataset_name_template = "dtype:{dtype}"

    # translation from pandas object dtype to strict string dtype is necessary
    obj_dtype = numpy.dtype('O')
    string_dtype = h5py.string_dtype()

    def dump(self, data, key):
        seed, group_name, name = self.resolve_key(key)
        with h5py.File(self.hdf5(seed), "a") as container:
            # version file. should we ever update the format it will make life easier
            container.attrs[self.file_version_attr] = self.file_version_id
            if group_name in container:
                raise LookupError(f"Cannot dump data for key {key} - would overwrite")
            group = container.create_group(group_name)
            self.save_df_to_group(group, name, data)

    def load(self, key):
        seed, group_name, name = self.resolve_key(key)
        with h5py.File(self.hdf5(seed), "r") as container:
            assert container.attrs[self.file_version_attr] == self.file_version_id, 'Unknown file version'
            try:
                group = container.get(group_name)
                return self.load_df_from_group(group, name)
            except KeyError:
                raise RuntimeError(f"No data set for {key} saved!")

    def resolve_key(self, key):
        if key.data_type in DataTypes.DataFrame_types:
            seed = key.seed
            group = key.data_type
            name = key.key
        else:
            msg = f"Invalid 'type' of data: {key.data_type}"
            raise ValueError(msg)
        return seed, group, name

    # non-interface methods
    @classmethod
    def from_paths(cls, paths: Paths):
        hdf5_path = str(paths.root_dir / "seiir-data-{{}}.hdf5")
        return cls(hdf5_path)

    def __init__(self, hdf5_template):  # TODO: type annotate
        self.hdf5_template = hdf5_template

    def hdf5(self, seed: str):
        return self.hdf5_template.format(seed)

    def save_df_to_group(self, parent_group, name, df):
        """
        It turns out hdf files use a single dtype for the entire dataset. This
        is reasonable, but makes more work for us.
        """
        # because we store 1 object and not many we must nest
        # use `name` for the nested directory-like thing
        group = parent_group.create_group(name)
        # determine groupings of data by common dtype
        by_dtype = defaultdict(list)
        for col_name, df_dtype in df.dtypes.iteritems():
            # object dtype is a no-go. cast to string
            if df_dtype == self.obj_dtype:
                assert isinstance(df[col_name].iloc[0], str), f"Column {col_name} is non-str objects - cannot save!"
                dt = self.string_dtype
            else:
                dt = df_dtype
            by_dtype[dt].append(col_name)

        # store data by common dtype to reduce number of datasets. note order
        order = []
        for dtype, column_names in by_dtype.items():
            dtype_dataset_name = numpy.string_(self.dataset_name_template.format(dtype=dtype))
            order.append(dtype_dataset_name)

            # TODO: enable compression
            # https://docs.h5py.org/en/2.6.0/high/dataset.html#filter-pipeline
            ds = group.create_dataset(dtype_dataset_name, data=df[column_names], dtype=dtype)
            # make data self-describing by assigning column names
            # https://docs.h5py.org/en/2.6.0/high/attr.html#attributes
            ds.attrs[self.column_names_ds_attr] = [numpy.string_(c) for c in column_names]

        # store metadata - all the column names and the load order
        group.create_dataset(self.load_order_ds_name, data=order, dtype=self.string_dtype)
        group.create_dataset(self.column_names_order,
                             data=[numpy.string_(c) for c in df.columns],
                             dtype=self.string_dtype)

    def load_df_from_group(self, parent_group, name):
        group = parent_group[name]
        # providing a "name" key to a group gets you a Dataset
        # to access the Dataset values, index with an empty tuple
        # https://docs.h5py.org/en/2.6.0/high/dataset.html#reading-writing-data
        # TODO: investigate if we can load more efficiently e.g., stream from
        # the Dataset directly into the pandas DataFrame
        ds_to_load = group[self.load_order_ds_name][()]

        df_pieces = []
        for dataset_name in ds_to_load:
            ds = group[dataset_name]
            # convert bytes to strings
            column_names = [X.decode() for X in ds.attrs[self.column_names_ds_attr]]
            arr = ds[()]
            df_piece = pandas.DataFrame(arr, columns=column_names)
            df_pieces.append(df_piece)

        result = pandas.concat(df_pieces, axis=1)

        # re-order columns to exact order they were saved in
        saved_col_order = group[self.column_names_order][()]
        result = result[saved_col_order]

        return result
