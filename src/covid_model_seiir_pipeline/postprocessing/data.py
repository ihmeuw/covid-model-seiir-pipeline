import pandas as pd

from covid_model_seiir_pipeline import io
from covid_model_seiir_pipeline.postprocessing.specification import PostprocessingSpecification


class PostprocessingDataInterface:

    def __init__(self,
                 forecast_root: io.ForecastRoot,
                 postprocessing_root: io.PostprocessingRoot):
        self.forecast_root = forecast_root
        self.postprocessing_root = postprocessing_root

    @classmethod
    def from_specification(cls, specification: PostprocessingSpecification):
        forecast_root = io.ForecastRoot(specification.data.forecast_version)
        postprocessing_root = io.PostprocessingRoot(specification.data.output_root)

        return cls(
            forecast_root=forecast_root,
            postprocessing_root=postprocessing_root,
        )

    def make_dirs(self, **prefix_args):
        io.touch(self.postprocessing_root, **prefix_args)

    ################
    # Data loaders #
    ################

    def load_resampling_map(self):
        return io.load(self.postprocessing_root.resampling_map())

    def load_raw_covariates(self, scenario: str, draw_id: int):
        covariates = io.load(self.forecast_root.raw_covariates(scenario=scenario, draw_id=draw_id))
        covariates['date'] = pd.to_datetime(covariates['date'])
        return covariates

    def load_components(self, scenario: str, draw_id: int) -> pd.DataFrame:
        components = io.load(self.forecast_root.component_draws(scenario=scenario, draw_id=draw_id))
        components['date'] = pd.to_datetime(components['date'])
        return components.set_index(['location_id', 'date'])

    def load_beta_scales(self, scenario: str, draw_id: int):
        return io.load(self.forecast_root.beta_scaling(scenario=scenario, draw_id=draw_id))

    def load_raw_outputs(self, scenario: str, draw_id: int) -> pd.DataFrame:
        return io.load(self.forecast_root.raw_outputs(scenario=scenario, draw_id=draw_id))

    ################
    # Data writers #
    ################

    def save_resampling_map(self, resampling_map):
        io.dump(resampling_map, self.postprocessing_root.resampling_map())

    def save_output_draws(self, output_draws: pd.DataFrame, scenario: str, measure: str):
        io.dump(output_draws, self.postprocessing_root.output_draws(scenario=scenario, measure=measure))

    def save_output_summaries(self, output_summaries: pd.DataFrame, scenario: str, measure: str):
        io.dump(output_summaries, self.postprocessing_root.output_summaries(scenario=scenario, measure=measure))

    def save_output_miscellaneous(self, output_miscellaneous: pd.DataFrame, scenario: str, measure: str):
        io.dump(output_miscellaneous, self.postprocessing_root.output_miscellaneous(scenario=scenario, measure=measure))
