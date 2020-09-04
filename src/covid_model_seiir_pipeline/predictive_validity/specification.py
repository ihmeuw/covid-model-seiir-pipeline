from dataclasses import dataclass
from typing import Dict, Tuple, List

from covid_model_seiir_pipeline.utilities import Specification, asdict


@dataclass
class HoldoutVersionSpecification:
    """Regression specification for a covariate."""

    # model params
    holdout_days: int
    infectionator_version: str
    covariate_version: str

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists.

        Drops the name parameter as it's used as a key in the specification.

        """
        return {k: v for k, v in asdict(self).items() if k != 'name'}


class PredictiveValiditySpecification(Specification):
    """Specification for a regression run."""

    def __init__(self,
                 output_root: str,
                 forecast_scenario: str,
                 holdout_versions: List[HoldoutVersionSpecification],
                 alphas: List[List[float]],
                 thetas: List[float],
                 beta_scaling_average_over_maxes: List[int]):
        self._output_root = output_root
        self._forecast_scenario = forecast_scenario
        self._holdout_versions = holdout_versions
        self._alphas = alphas
        self._thetas = thetas
        self._beta_scaling_average_over_maxes = beta_scaling_average_over_maxes

    @classmethod
    def parse_spec_dict(cls, predictive_validity_spec_dict: Dict) -> Tuple:
        """Constructs a regression specification from a dictionary."""
        output_root = predictive_validity_spec_dict.get('output_root', '')
        if not output_root:
            raise ValueError('Must provide an output root.')

        holdout_version_dicts = predictive_validity_spec_dict.get('holdout_version', [])
        if not holdout_version_dicts:
            raise ValueError('Must have at least one holdout version.')
        holdout_versions = [HoldoutVersionSpecification(**holdout_version) for holdout_version in holdout_version_dicts]
        alphas = predictive_validity_spec_dict.get('alpha', [[0.9, 1.0]])
        thetas = predictive_validity_spec_dict.get('theta', [0.0])
        beta_scaling_average_over_maxes = predictive_validity_spec_dict.get(
            'beta_scaling_average_over_max', [42]
        )

        return output_root, holdout_versions, alphas, thetas, beta_scaling_average_over_maxes

    @property
    def output_root(self) -> str:
        return self._output_root

    @property
    def forecast_scenario(self) -> str:
        return self._forecast_scenario

    @property
    def holdout_versions(self) -> List[HoldoutVersionSpecification]:
        return self._holdout_versions[:]

    @property
    def alphas(self) -> List[List[float]]:
        return self._alphas[:]

    @property
    def thetas(self) -> List[float]:
        return self._thetas[:]

    @property
    def beta_scaling_average_over_maxes(self) -> List[int]:
        return self._beta_scaling_average_over_maxes[:]

    def to_dict(self) -> Dict:
        """Converts the specification to a dict."""
        spec = {
            'output_root': self.output_root,
            'forecast_scenario': self.forecast_scenario,
            'holdout_versions': [h.to_dict() for h in self.holdout_versions],
            'alpha': self.alphas,
            'theta': self.thetas,
            'beta-scaling_average_over_maxes': self.beta_scaling_average_over_maxes,
        }
        return spec
