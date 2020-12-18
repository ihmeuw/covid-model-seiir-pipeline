"""Concrete representations of on disk data sources."""
import itertools
from pathlib import Path
from typing import List, Union

from .keys import (
    DatasetType,
    MetadataType,
    DatasetKey,
    LEAF_TEMPLATES,
    PREFIX_TEMPLATES,
)
from .marshall import (
    DATA_STRATEGIES,
    METADATA_STRATEGIES,
)


class DataRoot:
    """Representation of a version of data from a particular source or sink.

    This class provide convenience methods for the I/O tooling to inspect
    the structure of a data source or sink as defined by its subclasses.

    Subclasses are responsible for declaring their structure by assigning
    :class:`DatasetType` and :class:`MetadataType` class variables. Instances
    of subclasses then serve as factories for generating keys pointing
    to particular data sets which can be used to transfer data and metadata
    to and from disk.

    Parameters
    ----------
    root
        An existing directory on disk where data will be read from or written
        to.
    data_format
        The on disk format for data sets in the data root.
    metadata_format
        The on disk format for metadata in the data root.

    """

    def __init__(self, root: Union[str, Path], data_format: str = 'csv', metadata_format: str = 'yaml'):
        self._root = Path(root)
        if data_format not in DATA_STRATEGIES:
            raise ValueError(f'Invalid data format {data_format} for {type(self).__name__}. '
                             f'Valid data formats are {list(DATA_STRATEGIES)}.')
        self._data_format = data_format
        if metadata_format not in METADATA_STRATEGIES:
            raise ValueError(f'Invalid metadata format {metadata_format} for {type(self).__name__}. '
                             f'Valid data formats are {list(METADATA_STRATEGIES)}.')
        self._metadata_format = metadata_format

    @property
    def dataset_types(self) -> List[str]:
        """A list of all named dataset types in the data root."""
        return [dataset_name for dataset_name, attr in type(self).__dict__.items()
                if isinstance(attr, DatasetType)]

    @property
    def metadata_types(self) -> List[str]:
        """A list of all named metadata types in the data root."""
        return [metadata_name for metadata_name, attr in type(self).__dict__.items()
                if isinstance(attr, MetadataType)]

    def terminal_paths(self, **prefix_args: List[str]) -> List[Path]:
        """Resolves and returns all terminal directory and container paths.

        Parameters
        ----------
        prefix_args
            Lists of concrete prefix fill values provided as keyword arguments
            where the keyword is a format value in a prefix template.

        Returns
        -------
            A list of terminal paths without extensions to be interpreted
            by an I/O strategy.

        """
        paths = []
        dataset_types = [dataset_type for dataset_type in type(self).__dict__.values()
                         if isinstance(dataset_type, DatasetType)]
        for dataset_type in dataset_types:
            if dataset_type.prefix_template is not None:
                for arg_set in itertools.product(*prefix_args.values()):
                    prefix_template_kwargs = dict(zip(prefix_args.keys(), arg_set))
                    prefix = dataset_type.prefix_template.format(**prefix_template_kwargs)
                    path = self._root / prefix / dataset_type.name
                    paths.append(path)
            else:
                path = self._root / dataset_type.name
                paths.append(path)
        paths = list(set(paths))  # Drop duplicates
        return paths


class InfectionRoot(DataRoot):
    """Data root representing infectionator outputs."""
    metadata = MetadataType('metadata')

    def modeled_locations(self) -> List[int]:
        """Retrieve all of the location specific infection directories."""
        return [int(p.name.split('_')[-1]) for p in self._root.iterdir() if p.is_dir()]

    def infections(self, location_id: int, draw_id: int):
        """Hack around infectionator file layout to provide a consistent
        interface."""
        data_type = str([m for m in self._root.glob(f"*_{location_id}")][0])
        return DatasetKey(
            root=self._root,
            disk_format=self._data_format,
            data_type=data_type,
            leaf_name=f'draw{draw_id:04}_prepped_deaths_and_cases_all_age',
            prefix=None,
        )


class CovariateRoot(DataRoot):
    """Data root representing prepped covariates."""
    metadata = MetadataType('metadata')

    air_pollution_pm_2_5 = DatasetType('air_pollution_pm_2_5', LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE)
    lri_mortality = DatasetType('lri_mortality', LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE)
    mask_use = DatasetType('mask_use', LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE)
    mobility = DatasetType('mobility', LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE)
    pneumonia = DatasetType('pneumonia', LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE)
    proportion_over_2_5k = DatasetType('proportion_over_2_5k', LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE)
    proportion_under_100m = DatasetType('proportion_under_100m', LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE)
    smoking_prevalence = DatasetType('smoking_prevalence', LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE)
    testing = DatasetType('testing', LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE)

    mobility_info = DatasetType('mobility', LEAF_TEMPLATES.COV_INFO_TEMPLATE)
    vaccine_info = DatasetType('vaccine_coverage', LEAF_TEMPLATES.COV_INFO_TEMPLATE)

    # Getters provide dynamic keys to support experimentation with custom covariates.
    def __getattr__(self, item) -> DatasetType:
        setattr(type(self), item, DatasetType(item, LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE))
        return getattr(self, item)

    def __getitem__(self, item) -> DatasetType:
        return getattr(self, item)


class RegressionRoot(DataRoot):
    """Data root representing regression stage outputs."""
    metadata = MetadataType('metadata')
    specification = MetadataType('regression_specification')
    locations = MetadataType('locations')

    beta = DatasetType('beta', LEAF_TEMPLATES.DRAW_TEMPLATE)
    coefficients = DatasetType('coefficients', LEAF_TEMPLATES.DRAW_TEMPLATE)
    data = DatasetType('data', LEAF_TEMPLATES.DRAW_TEMPLATE)
    dates = DatasetType('dates', LEAF_TEMPLATES.DRAW_TEMPLATE)
    parameters = DatasetType('parameters', LEAF_TEMPLATES.DRAW_TEMPLATE)


class ForecastRoot(DataRoot):
    """Data root representing forecast stage outputs."""
    metadata = MetadataType('metadata')
    specification = MetadataType('forecast_specification')
    resampling_map = MetadataType('resampling_map')

    beta_scaling = DatasetType('beta_scaling', LEAF_TEMPLATES.DRAW_TEMPLATE, PREFIX_TEMPLATES.SCENARIO_TEMPLATE)
    component_draws = DatasetType('component_draws', LEAF_TEMPLATES.DRAW_TEMPLATE, PREFIX_TEMPLATES.SCENARIO_TEMPLATE)
    raw_covariates = DatasetType('raw_covariates', LEAF_TEMPLATES.DRAW_TEMPLATE, PREFIX_TEMPLATES.SCENARIO_TEMPLATE)
    raw_outputs = DatasetType('raw_outputs', LEAF_TEMPLATES.DRAW_TEMPLATE, PREFIX_TEMPLATES.SCENARIO_TEMPLATE)


class PostprocessingRoot(DataRoot):
    """Data root representing postprocessing stage outputs."""
    metadata = MetadataType('metadata')
    specification = MetadataType('postprocessing_specification')

    output_draws = DatasetType('output_draws',
                               LEAF_TEMPLATES.MEASURE_TEMPLATE,
                               PREFIX_TEMPLATES.SCENARIO_TEMPLATE)
    output_summaries = DatasetType('output_summaries',
                                   LEAF_TEMPLATES.MEASURE_TEMPLATE,
                                   PREFIX_TEMPLATES.SCENARIO_TEMPLATE)
    output_miscellaneous = DatasetType('output_miscellaneous',
                                       LEAF_TEMPLATES.MEASURE_TEMPLATE,
                                       PREFIX_TEMPLATES.SCENARIO_TEMPLATE)
