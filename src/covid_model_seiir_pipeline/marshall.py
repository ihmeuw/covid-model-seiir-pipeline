"""
Code for marshalling data (currently only DataFrames) to/from disk.

This module holds 2 key concepts meant to be used externally - Keys and Marshall classes.

Keys are used to identify data types and how they are stored. This is by
embedding related information with them (draw_id for regression and location_id
for forecast).

Marshall classes come in several flavors named after the related storage backend.
Marshall classes should created via Marshall.from_paths(paths_cls) until Paths
are fully deprecated and removed.
"""
from collections import defaultdict
from contextlib import contextmanager
import io
from pathlib import Path
import typing
import zipfile

import h5py
import numpy
import pandas

from covid_model_seiir_pipeline.paths import (
    Paths,
    DRAW_FILE_TEMPLATE,
    MEASURE_FILE_TEMPLATE
)
from covid_shared.shell_tools import mkdir


class DataTypes:
    """
    Enumerations of data types as understood by the Marshall interface.

    These enumerations can be expected to appear as sub-directories or other
    directory-like separator names in data stored by Marshall instances.
    """
    beta_scales = "beta_scaling"
    coefficient = "coefficients"
    components = "component_draws"
    date = "dates"
    forecast_raw_outputs = "raw_outputs"
    forecast_raw_covariates = "raw_covariates"
    forecast_output_draws = "output_draws"
    forecast_output_summaries = "output_summaries"
    forecast_output_miscellaneous = "output_miscellaneous"
    reimposition_dates = "reimposition_dates"
    location_data = "data"
    parameter = "parameters"
    regression_beta = "beta"

    # types which serialize to a DataFrame
    DataFrame_types = frozenset([
        location_data,
        parameter,
        date,
        coefficient,
        regression_beta,
        beta_scales,
        components,
        forecast_raw_outputs,
        forecast_raw_covariates,
        forecast_output_draws,
        forecast_output_summaries,
        forecast_output_miscellaneous,
        reimposition_dates
    ])


# TODO: swap DRAW_FILE_TEMPLATE in keys with a generic DRAW_TEMPLATE and
#  let the marshalls worry about extensions.
# TODO: consider a factory for keys that takes in the template to use so that
#  the specification of file layout can be done at a higher level of
#  abstraction.

class Keys:
    """
    Identifies data payload types and stores related information.

    Callers should never use init and instead use one of the many factory
    methods which embed important related information to the variable
    draw_id/location_id argument.
    """
    def __init__(self, data_type, template, **key_args):
        self.data_type = data_type
        self.template = template
        self.key_args = key_args

    # factories related to regression
    @classmethod
    def coefficient(cls, draw_id):
        return cls(DataTypes.coefficient, DRAW_FILE_TEMPLATE, draw_id=draw_id)

    @classmethod
    def date(cls, draw_id):
        return cls(DataTypes.date, DRAW_FILE_TEMPLATE, draw_id=draw_id)

    @classmethod
    def location_data(cls, draw_id):
        return cls(DataTypes.location_data, DRAW_FILE_TEMPLATE, draw_id=draw_id)

    @classmethod
    def parameter(cls, draw_id):
        return cls(DataTypes.parameter, DRAW_FILE_TEMPLATE, draw_id=draw_id)

    @classmethod
    def regression_beta(cls, draw_id):
        return cls(DataTypes.regression_beta, DRAW_FILE_TEMPLATE, draw_id=draw_id)

    # factories related to forecasting
    @classmethod
    def beta_scales(cls, scenario, draw_id):
        return cls(DataTypes.beta_scales, DRAW_FILE_TEMPLATE, scenario=scenario, draw_id=draw_id)

    @classmethod
    def components(cls, scenario, draw_id):
        return cls(DataTypes.components, DRAW_FILE_TEMPLATE, scenario=scenario, draw_id=draw_id)

    @classmethod
    def forecast_raw_outputs(cls, scenario, draw_id):
        return cls(DataTypes.forecast_raw_outputs, DRAW_FILE_TEMPLATE, scenario=scenario, draw_id=draw_id)

    @classmethod
    def forecast_raw_covariates(cls, scenario, draw_id):
        return cls(DataTypes.forecast_raw_covariates, DRAW_FILE_TEMPLATE, scenario=scenario, draw_id=draw_id)

    @classmethod
    def forecast_output_draws(cls, scenario, measure):
        return cls(DataTypes.forecast_output_draws, MEASURE_FILE_TEMPLATE, scenario=scenario, measure=measure)

    @classmethod
    def forecast_output_summaries(cls, scenario, measure):
        return cls(DataTypes.forecast_output_summaries, MEASURE_FILE_TEMPLATE, scenario=scenario, measure=measure)

    @classmethod
    def forecast_output_miscellaneous(cls, scenario, measure):
        return cls(DataTypes.forecast_output_miscellaneous, MEASURE_FILE_TEMPLATE, scenario=scenario, measure=measure)

    @classmethod
    def reimposition_dates(cls, scenario, reimposition_number):
        return cls(DataTypes.reimposition_dates, 'reimposition_{reimposition_number}',
                   scenario=scenario, reimposition_number=reimposition_number)

    # other methods/properties
    @property
    def key(self):
        return self.template.format(**self.key_args)

    def nested_dirs(self):
        """
        Return any nesting structure required between the root save location
        and the actual file.

        For regression tasks this is just the data_type. For forecasting tasks
        this is the scenario and the data type.
        """
        if 'scenario' in self.key_args:
            return "/".join([self.key_args['scenario'], self.data_type])
        else:
            return self.data_type

    @property
    def seed(self):
        """
        Returns a seed value for enabling concurrency.

        This is a sort of hack - we embed the concurrency story into the Keys
        class which can then inform other things.
        """
        # everything is done by draw_id, so always use that as the seed value
        return "draw-{draw_id}".format(**self.key_args)

    def __repr__(self):
        return f"Keys({self.data_type!r}, {self.template!r}, **{self.key_args!r})"


# TODO: Abstract an interface for use in typing.
# TODO: Strategy pattern for selecting marshalls from a combination of
#  run specification and inference on input data.

class CSVMarshall:
    """
    Marshalls DataFrames to/from CSV files.

    This implementation directly mirrors existing behavior but does so within a
    new marshalling interface.
    """
    # interface methods
    def dump(self, data: pandas.DataFrame, key, strict=True):
        path = self.resolve_key(key)
        if not path.parent.is_dir():
            mkdir(path.parent, parents=True)
        else:
            if strict and path.exists():
                msg = f"Cannot dump data for key {key} - would overwrite"
                raise LookupError(msg)

        data.to_csv(path, index=False)

    def load(self, key):
        path = self.resolve_key(key)
        return pandas.read_csv(path)

    def resolve_key(self, key):
        if key.data_type in DataTypes.DataFrame_types:
            path = (self.root / key.nested_dirs() / key.key).with_suffix(".csv")
        else:
            msg = f"Invalid 'type' of data: {key.data_type}"
            raise ValueError(msg)

        return path

    @classmethod
    def from_paths(cls, paths: Paths):
        return cls(paths.root_dir)

    # non-interface methods
    def __init__(self, root: Path):
        self.root = root


class ZipMarshall:
    # interface methods
    def dump(self, data: pandas.DataFrame, key, strict=True):
        if not strict:
            raise NotImplementedError

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
            seed = key.seed
            path = f"{key.nested_dirs()}/{key.key}.csv"
        else:
            msg = f"Invalid 'type' of data: {key.data_type}"
            raise ValueError(msg)

        return seed, path

    @classmethod
    def from_paths(cls, paths: Paths):
        # technically unsafe - {} values in the root_dir could break interpolation
        zip_path = str(paths.root_dir / "seiir-data-{{}}.zip")
        return cls(zip_path)

    # non-interface methods
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

    def dump(self, data, key, strict=True):
        if not strict:
            raise NotImplementedError

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
            group = key.nested_dirs()
            name = key.key
        else:
            msg = f"Invalid 'type' of data: {key.data_type}"
            raise ValueError(msg)
        return seed, group, name

    @classmethod
    def from_paths(cls, paths: Paths):
        hdf5_path = str(paths.root_dir / "seiir-data-{{}}.hdf5")
        return cls(hdf5_path)

    # non-interface methods
    def __init__(self, hdf5_template):  # TODO: type annotate
        self.hdf5_template = hdf5_template
        self.dtype_marshall = DtypeMarshall()

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
            by_dtype[df_dtype].append(col_name)

        # store data by common dtype to reduce number of datasets. note order
        order = []
        for pd_dtype, column_names in by_dtype.items():
            data, dtype, dt_label = self.dtype_marshall.to_array(df[column_names])

            dtype_dataset_name = numpy.string_(self.dataset_name_template.format(dtype=dt_label))
            order.append(dtype_dataset_name)

            # TODO: enable compression
            # https://docs.h5py.org/en/2.6.0/high/dataset.html#filter-pipeline
            ds = group.create_dataset(dtype_dataset_name, data=data, dtype=dtype)

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
        for columns_by_type in ds_to_load:
            ds = group[columns_by_type]
            dtype: str = columns_by_type[6:].decode()  # chomp 'dtype:' prefix and convert from bytes
            # convert bytes to strings
            column_names = [X.decode() for X in ds.attrs[self.column_names_ds_attr]]
            arr = ds[()]
            df_piece = self.dtype_marshall.to_pandas(arr, dtype, columns=column_names)
            df_pieces.append(df_piece)

        result = pandas.concat(df_pieces, axis=1)

        # re-order columns to exact order they were saved in
        saved_col_order = group[self.column_names_order][()]
        saved_col_order = self.dtype_marshall.from_bytes(saved_col_order)
        result = result[saved_col_order]

        return result


# TODO: Consider renaming. Totally different kind of marshalling
#    happening here.

class DtypeMarshall:
    """
    Handles the odds and ends of marshalling between pandas and h5py dtypes.
    """
    dt_epoch = pandas.Timestamp("1970-01-01")
    dt_res = pandas.Timedelta("1s")

    dt_map = {
        # dtype -> (result_dt, hdf_dt, label)
        numpy.datetime64: (numpy.int64, numpy.int64, "datetime64"),
        # we cannot use numpy.string_ here because it truncates
        numpy.object_: (numpy.object_, h5py.string_dtype(), "string"),
    }

    def to_array(self, df: pandas.DataFrame) -> typing.Tuple[numpy.array, numpy.dtype, str]:
        try:
            dt, = df.dtypes.unique()
        except ValueError:
            types = df.dtypes.unique()
            raise TypeError(f"to_array must provide a DataFrame with a single dtype - have {types}")

        tmp = self.dt_map.get(dt.type)
        if tmp is None:
            return df.values, dt.type, dt.type.__name__  # no conversion necessary
        else:
            result_dt, out_dt, label = tmp

        result = numpy.empty(df.shape, dtype=result_dt)
        for i, name in enumerate(df.columns):
            converted = self.as_hdf_dtype(df[name])
            result[:, i] = converted

        return result, out_dt, label

    def to_pandas(self, a: numpy.array, dtype_str: str, columns: typing.List[str]) -> pandas.DataFrame:
        # this indexing looks backwards but is not
        return pandas.DataFrame({column: self.as_pandas_dtype(a[:, i], dtype_str)
                                 for i, column in enumerate(columns)})

    def as_hdf_dtype(self, s: pandas.Series) -> numpy.array:
        t = s.dtype.type
        if issubclass(t, numpy.number):
            return s.values
        elif t is numpy.datetime64:
            return self.to_unix_epoch(s)
        elif t is numpy.object_:
            if not isinstance(s[0], str):
                t = type(s[0])
                msg = f"DataFrame columns with type object must be strings. Have {t}"
                raise TypeError(msg)
            return numpy.string_(s)
        else:
            raise NotImplementedError(f"No support to convert {t}")

    def as_pandas_dtype(self, a: numpy.array, dtype: str) -> pandas.Series:
        """
        Return array as Series.

        Note: strings are automagically converted back by h5py
        """
        if dtype == "datetime64":
            return self.from_unix_epoch(a)
        elif dtype == "string":
            return self.from_bytes(a)
        else:
            return pandas.Series(a)

    # datetime64 conversion
    def to_unix_epoch(self, s: pandas.Series) -> numpy.array:
        """
        Convert datetime-like value to Unix style timestamp.

        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#from-timestamps-to-epoch
        """
        # Note floor division. Guard against loss of information
        if s.dt.microsecond.any():
            raise NotImplementedError("No support for timestamps with sub-second values: {val}")
        # floor division forces int64 instead of float64. This is important!
        return (((s - self.dt_epoch) // self.dt_res).values)

    def from_unix_epoch(self, a: numpy.array) -> pandas.Series:
        """
        Convert Unix style timestamp to datetime-like value used by pandas.

        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#epoch-timestamps
        """
        return pandas.Series(pandas.to_datetime(a, unit='s'))

    def from_bytes(self, a: numpy.array) -> pandas.Series:
        """Convert array of bytes to utf-8 strings."""
        return pandas.Series(a).str.decode('utf-8')
