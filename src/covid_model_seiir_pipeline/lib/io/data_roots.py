"""Concrete representations of on disk data sources."""
import itertools
from pathlib import Path
from typing import List, Union

from covid_model_seiir_pipeline.lib.io.keys import (
    DatasetType,
    MetadataType,
    LEAF_TEMPLATES,
    PREFIX_TEMPLATES,
)
from covid_model_seiir_pipeline.lib.io.marshall import (
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
                    if dataset_type.leaf_template is not None:
                        path = self._root / prefix / dataset_type.name
                    else:
                        path = self._root / prefix
                    paths.append(path)
            else:
                if dataset_type.leaf_template is not None:
                    path = self._root / dataset_type.name
                else:
                    path = self._root
                paths.append(path)
        paths = list(set(paths))  # Drop duplicates
        return paths


###############
# Input Roots #
###############

class ModelInputsRoot(DataRoot):
    metadata = MetadataType('metadata')

    population = DatasetType('output_measures/population/all_populations')
    hospital_census = DatasetType('output_measures/hospitalizations/population')
    hospital_capacity = DatasetType('hospital_capacity')
    icu_census = DatasetType('output_measures/icu/population')
    serology = DatasetType('serology', LEAF_TEMPLATES.MEASURE_TEMPLATE)
    full_data = DatasetType('full_data_unscaled')
    full_data_extra_hospital = DatasetType('use_at_your_own_risk/full_data_extra_hospital')
    gbd_covariate = DatasetType('gbd_covariates', LEAF_TEMPLATES.MEASURE_TEMPLATE)


class AgeSpecificRatesRoot(DataRoot):
    metadata = MetadataType('metadata')

    rates_data = DatasetType(LEAF_TEMPLATES.MEASURE_TEMPLATE)


class MortalityScalarsRoot(DataRoot):
    metadata = MetadataType('metadata')

    total_covid_draw = DatasetType('total_covid_draw')
    total_covid_mean = DatasetType('total_covid_mean')


class MaskUseRoot(DataRoot):
    metadata = MetadataType('metadata')

    mask_use_data = DatasetType(LEAF_TEMPLATES.MEASURE_TEMPLATE)


class MobilityRoot(DataRoot):
    metadata = MetadataType('metadata')

    mobility_data = DatasetType(LEAF_TEMPLATES.MEASURE_TEMPLATE)


class PneumoniaRoot(DataRoot):
    metadata = MetadataType('metadata')

    pneumonia_data = DatasetType('pneumonia')


class PopulationDensityRoot(DataRoot):
    metadata = MetadataType('metadata')

    population_density_data = DatasetType('all_outputs_2020_full')


class TestingRoot(DataRoot):
    metadata = MetadataType('metadata')

    testing_data = DatasetType('forecast_raked_test_pc_simple')


class VariantPrevalenceRoot(DataRoot):
    metadata = MetadataType('metadata')

    prevalence = DatasetType(LEAF_TEMPLATES.VARIANT_SCENARIO)
    original_data = DatasetType('original_data', LEAF_TEMPLATES.MEASURE_TEMPLATE)


class VaccineCoverageRoot(DataRoot):
    metadata = MetadataType('metadata')

    brand_specific_coverage = DatasetType(LEAF_TEMPLATES.MEASURE_TEMPLATE)
    old_vaccine_coverage = DatasetType('slow_scenario_vaccine_coverage')
    series_hesitancy = DatasetType('time_series_vaccine_hesitancy')
    point_hesitancy = DatasetType('time_point_vaccine_hesitancy')


class VaccineEfficacyRoot(DataRoot):
    metadata = MetadataType('metadata')

    efficacy_table = DatasetType('vaccine_efficacy_table')
    waning_distribution = DatasetType(LEAF_TEMPLATES.MEASURE_TEMPLATE)


class CovariatePriorsRoot(DataRoot):
    metadata = MetadataType('metadata')

    priors = DatasetType('priors')


########################
# Pipeline Stage Roots #
########################

class PreprocessingRoot(DataRoot):
    metadata = MetadataType('metadata')
    specification = MetadataType('preprocessing_specification')

    hierarchy = DatasetType('hierarchy', LEAF_TEMPLATES.MEASURE_TEMPLATE)
    population = DatasetType('population', LEAF_TEMPLATES.MEASURE_TEMPLATE)
    reported_epi_data = DatasetType('reported_epi_data')
    age_patterns = DatasetType('age_patterns')
    total_covid_scalars = DatasetType('total_covid_scalars')
    seroprevalence = DatasetType('seroprevalence')
    seroprevalence_samples = DatasetType('seroprevalence_samples', LEAF_TEMPLATES.DRAW_TEMPLATE)
    sensitivity = DatasetType('sensitivity')
    sensitivity_samples = DatasetType('sensitivity_samples', LEAF_TEMPLATES.DRAW_TEMPLATE)
    testing_for_idr = DatasetType('testing')
    variant_prevalence = DatasetType('variant_prevalence', LEAF_TEMPLATES.VARIANT_SCENARIO)
    waning_parameters = DatasetType(LEAF_TEMPLATES.MEASURE_TEMPLATE)
    vaccine_summary = DatasetType('vaccine_summary')
    vaccine_uptake = DatasetType('vaccine_uptake', LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE)
    vaccine_risk_reduction = DatasetType('vaccine_risk_reduction', LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE)
    antiviral_coverage = DatasetType('antiviral_coverage', LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE)

    mask_use = DatasetType('mask_use', LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE, PREFIX_TEMPLATES.COVARIATES)
    mobility = DatasetType('mobility', LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE, PREFIX_TEMPLATES.COVARIATES)
    mobility_info = DatasetType('mobility', LEAF_TEMPLATES.COV_INFO_TEMPLATE, PREFIX_TEMPLATES.COVARIATES)
    pneumonia = DatasetType('pneumonia', LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE, PREFIX_TEMPLATES.COVARIATES)
    proportion_over_2_5k = DatasetType('proportion_over_2_5k', LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE, PREFIX_TEMPLATES.COVARIATES)
    testing = DatasetType('testing', LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE, PREFIX_TEMPLATES.COVARIATES)

    # Getters provide dynamic keys to support experimentation with custom covariates.
    def __getattr__(self, item: str) -> DatasetType:
        setattr(type(self), item, DatasetType(item, LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE, PREFIX_TEMPLATES.COVARIATES))
        return getattr(self, item)

    def __getitem__(self, item: str) -> DatasetType:
        return getattr(self, item)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


class FitRoot(DataRoot):
    metadata = MetadataType('metadata')
    specification = MetadataType('fit_specification')
    covariate_options = MetadataType('covariate_options')
    draw_resampling = MetadataType('draw_resampling')

    fit_failures = DatasetType('fit_failures')
    fit_residuals = DatasetType('fit_residuals')

    beta = DatasetType('beta', LEAF_TEMPLATES.MEASURE_DRAW_TEMPLATE)
    input_epi_measures = DatasetType('input_epi_measures', LEAF_TEMPLATES.MEASURE_DRAW_TEMPLATE)
    posterior_epi_measures = DatasetType('posterior_epi_measures', LEAF_TEMPLATES.MEASURE_DRAW_TEMPLATE)
    rates = DatasetType('rates', LEAF_TEMPLATES.MEASURE_DRAW_TEMPLATE)
    rates_data = DatasetType('rates_data', LEAF_TEMPLATES.MEASURE_DRAW_TEMPLATE)
    antiviral_effectiveness = DatasetType('antiviral_effectiveness', LEAF_TEMPLATES.MEASURE_DRAW_TEMPLATE)
    compartments = DatasetType('compartments', LEAF_TEMPLATES.MEASURE_DRAW_TEMPLATE)
    ode_parameters = DatasetType('ode_parameters', LEAF_TEMPLATES.MEASURE_DRAW_TEMPLATE)
    phis = DatasetType('phis', LEAF_TEMPLATES.DRAW_TEMPLATE)
    seroprevalence = DatasetType('seroprevalence', LEAF_TEMPLATES.MEASURE_DRAW_TEMPLATE)
    summary = DatasetType('summary', LEAF_TEMPLATES.MEASURE_TEMPLATE)


class RegressionRoot(DataRoot):
    metadata = MetadataType('metadata')
    specification = MetadataType('regression_specification')
    locations = MetadataType('locations')

    hierarchy = DatasetType('hierarchy')

    beta = DatasetType('beta', LEAF_TEMPLATES.DRAW_TEMPLATE)
    coefficients = DatasetType('coefficients', LEAF_TEMPLATES.DRAW_TEMPLATE)
    hospitalizations = DatasetType('hospitalizations', LEAF_TEMPLATES.MEASURE_TEMPLATE)


class ForecastRoot(DataRoot):
    metadata = MetadataType('metadata')
    specification = MetadataType('forecast_specification')

    beta_scaling = DatasetType('beta_scaling', LEAF_TEMPLATES.DRAW_TEMPLATE, PREFIX_TEMPLATES.SCENARIO_TEMPLATE)
    beta_residual = DatasetType('beta_residual', LEAF_TEMPLATES.DRAW_TEMPLATE, PREFIX_TEMPLATES.SCENARIO_TEMPLATE)
    ode_params = DatasetType('ode_params', LEAF_TEMPLATES.DRAW_TEMPLATE, PREFIX_TEMPLATES.SCENARIO_TEMPLATE)
    component_draws = DatasetType('component_draws', LEAF_TEMPLATES.DRAW_TEMPLATE, PREFIX_TEMPLATES.SCENARIO_TEMPLATE)
    raw_covariates = DatasetType('raw_covariates', LEAF_TEMPLATES.DRAW_TEMPLATE, PREFIX_TEMPLATES.SCENARIO_TEMPLATE)
    raw_outputs = DatasetType('raw_outputs', LEAF_TEMPLATES.DRAW_TEMPLATE, PREFIX_TEMPLATES.SCENARIO_TEMPLATE)


class PostprocessingRoot(DataRoot):
    metadata = MetadataType('metadata')
    specification = MetadataType('postprocessing_specification')
    resampling_map = MetadataType('resampling_map')

    output_draws = DatasetType('output_draws',
                               LEAF_TEMPLATES.MEASURE_TEMPLATE,
                               PREFIX_TEMPLATES.SCENARIO_TEMPLATE)
    output_summaries = DatasetType('output_summaries',
                                   LEAF_TEMPLATES.MEASURE_TEMPLATE,
                                   PREFIX_TEMPLATES.SCENARIO_TEMPLATE)
    output_miscellaneous = DatasetType('output_miscellaneous',
                                       LEAF_TEMPLATES.MEASURE_TEMPLATE,
                                       PREFIX_TEMPLATES.SCENARIO_TEMPLATE)


class DiagnosticsRoot(DataRoot):
    metadata = MetadataType('metadata')
    specification = MetadataType('diagnostics_specification')


#######################
# Side analysis roots #
#######################

class OOSHoldoutRoot(DataRoot):
    metadata = MetadataType('metadata')
    specification = MetadataType('oos_holdout_specification')

    beta = DatasetType('beta', LEAF_TEMPLATES.DRAW_TEMPLATE)
    coefficients = DatasetType('coefficients', LEAF_TEMPLATES.DRAW_TEMPLATE)
    beta_scaling = DatasetType('beta_scaling', LEAF_TEMPLATES.DRAW_TEMPLATE)
    beta_residual = DatasetType('beta_residual', LEAF_TEMPLATES.DRAW_TEMPLATE)
    raw_oos_outputs = DatasetType('raw_oos_outputs', LEAF_TEMPLATES.DRAW_TEMPLATE)
    deltas = DatasetType('deltas', LEAF_TEMPLATES.DRAW_TEMPLATE)


class CounterfactualInputRoot(DataRoot):
    metadata = MetadataType('metadata')

    beta = DatasetType('beta', LEAF_TEMPLATES.DRAW_TEMPLATE, PREFIX_TEMPLATES.SCENARIO_TEMPLATE)
    vaccine_uptake = DatasetType('vaccine_uptake', LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE)
    etas = DatasetType('etas', LEAF_TEMPLATES.COV_SCENARIO_TEMPLATE)


class CounterfactualRoot(DataRoot):
    metadata = MetadataType('metadata')
    specification = MetadataType('counterfactual_specification')
    forecast_specification = MetadataType('forecast_specification')

    ode_params = DatasetType('ode_params', LEAF_TEMPLATES.DRAW_TEMPLATE, PREFIX_TEMPLATES.SCENARIO_TEMPLATE)
    component_draws = DatasetType('component_draws', LEAF_TEMPLATES.DRAW_TEMPLATE, PREFIX_TEMPLATES.SCENARIO_TEMPLATE)
    raw_outputs = DatasetType('raw_outputs', LEAF_TEMPLATES.DRAW_TEMPLATE, PREFIX_TEMPLATES.SCENARIO_TEMPLATE)
