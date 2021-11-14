from pathlib import Path
from typing import Any, List

from covid_input_seir_covariates.utilities import CovariateGroup

from . import (
    mask_use,
    mobility,
    pneumonia,
    population_density,
    testing,
    vaccine_coverage,
    variant_prevalence,
)
from .. import gbd_covariates


class CovariateLoader:
    def __init__(self,
                 # A covariate module with a COVARIATE_NAMES variable
                 # and a get_covariates method. Don't know how to type this.
                 loader: Any):
        self._loader = loader
        self._validate_loader_interface()

    def _validate_loader_interface(self):
        if not hasattr(self._loader, 'get_covariates'):
            raise ValueError(f"{self.name} does not define get_covariates method")
        if not hasattr(self._loader, "COVARIATE_NAMES"):
            raise ValueError(f"{self.name} does not define COVARIATE_NAMES attribute")
        if not hasattr(self._loader, "DEFAULT_OUTPUT_ROOT"):
            raise ValueError(f"{self.name} does not define DEFAULT_OUTPUT_ROOT attribute")

    @property
    def default_root(self) -> Path:
        return self._loader.DEFAULT_OUTPUT_ROOT

    @property
    def name(self):
        return self._loader.__name__.rsplit(".", 1)[1]

    @property
    def cli_arg_name(self) -> str:
        return self.name.replace('_', '-')

    @property
    def covariate_names(self) -> List[str]:
        return self._loader.COVARIATE_NAMES

    def __call__(self, covariates_root: Path) -> CovariateGroup:
        return self._loader.get_covariates(covariates_root)


covariate_loaders = (
    CovariateLoader(gbd_covariates),
    CovariateLoader(mask_use),
    CovariateLoader(mobility),
    CovariateLoader(pneumonia),
    CovariateLoader(population_density),
    CovariateLoader(testing),
    CovariateLoader(vaccine_coverage),
    CovariateLoader(variant_prevalence),
)

quiet_logging = (
    *gbd_covariates.COVARIATE_NAMES,
    *pneumonia.COVARIATE_NAMES,
    *population_density.COVARIATE_NAMES,
)
