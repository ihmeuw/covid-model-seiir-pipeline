from pathlib import Path
from typing import List, Union

from .keys import (
    DatasetType,
    MetadataType,
    DatasetKey,
    LEAF_TEMPLATES,
    PATH_TEMPLATES,
)
from .marshall import (
    DATA_STRATEGIES,
    METADATA_STRATEGIES,
)


class DataRoot:

    def __init__(self, root: Union[str, Path], data_format: str = 'csv', metadata_format: str = 'yaml'):
        self._root = Path(root)
        if data_format not in DATA_STRATEGIES:
            raise
        self._data_format = data_format
        if metadata_format not in METADATA_STRATEGIES:
            raise
        self._metadata_format = metadata_format

    def dataset_types(self) -> List[str]:
        return [dataset_name for dataset_name, attr in type(self).__dict__.items()
                if isinstance(attr, DatasetType)]

    def metadata_types(self) -> List[str]:
        return [metadata_name for metadata_name, attr in type(self).__dict__.items()
                if isinstance(attr, MetadataType)]


class InfectionRoot(DataRoot):
    metadata = MetadataType('metadata')

    def modeled_locations(self) -> List[int]:
        """Retrieve all of the location specific infection directories."""
        return [int(p.name.split('_')[-1]) for p in self._root.iterdir() if p.is_dir()]

    @property
    def infections(self):
        def _infections(location_id: int, draw_id: int):
            data_type = str([m for m in self._root.glob(f"*_{location_id}")][0])
            return DatasetKey(
                root=self._root,
                disk_format=self._data_format,
                data_type=data_type,
                leaf_name=f'draw{draw_id:04}_prepped_deaths_and_cases_all_age.csv',
                path_name=None,
            )
        return _infections


class CovariateRoot(DataRoot):
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

    def __getattr__(self, item) -> DatasetType:
        setattr(type(self), item, DatasetType(item, LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE))
        return getattr(self, item)

    def __getitem__(self, item) -> DatasetType:
        return getattr(self, item)


class RegressionRoot(DataRoot):
    metadata = MetadataType('metadata')
    specification = MetadataType('regression_specification')
    locations = MetadataType('locations')

    beta = DatasetType('beta', LEAF_TEMPLATES.DRAW_TEMPLATE)
    coefficients = DatasetType('coefficients', LEAF_TEMPLATES.DRAW_TEMPLATE)
    data = DatasetType('data', LEAF_TEMPLATES.DRAW_TEMPLATE)
    dates = DatasetType('dates', LEAF_TEMPLATES.DRAW_TEMPLATE)
    parameters = DatasetType('parameters', LEAF_TEMPLATES.DRAW_TEMPLATE)


class ForecastRoot(DataRoot):
    metadata = MetadataType('metadata')
    specification = MetadataType('forecast_specification')
    resampling_map = MetadataType('resampling_map')

    beta_scaling = DatasetType('beta_scaling', LEAF_TEMPLATES.DRAW_TEMPLATE, PATH_TEMPLATES.SCENARIO_TEMPLATE)
    component_draws = DatasetType('component_draws', LEAF_TEMPLATES.DRAW_TEMPLATE, PATH_TEMPLATES.SCENARIO_TEMPLATE)
    raw_covariates = DatasetType('raw_covariates', LEAF_TEMPLATES.DRAW_TEMPLATE, PATH_TEMPLATES.SCENARIO_TEMPLATE)
    raw_outputs = DatasetType('raw_outputs', LEAF_TEMPLATES.DRAW_TEMPLATE, PATH_TEMPLATES.SCENARIO_TEMPLATE)

    # TODO: Move to postprocessing root
    output_draws = DatasetType('output_draws', LEAF_TEMPLATES.MEASURE_TEMPLATE, PATH_TEMPLATES.SCENARIO_TEMPLATE)
    output_summaries = DatasetType('output_summaries',
                                   LEAF_TEMPLATES.MEASURE_TEMPLATE, PATH_TEMPLATES.SCENARIO_TEMPLATE)
    output_miscellaneous = DatasetType('output_miscellaneous',
                                       LEAF_TEMPLATES.MEASURE_TEMPLATE, PATH_TEMPLATES.SCENARIO_TEMPLATE)
