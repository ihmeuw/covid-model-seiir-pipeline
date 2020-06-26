import pandas as pd
import numpy as np
from slime.model.cov_model import CovModelSet, CovModel
from covid_model_seiir_pipeline.ode_fit.model import ODEProcess
from covid_model_seiir_pipeline.regression.model import BetaRegressor, BetaRegressorSequential, predict
from covid_model_seiir_pipeline.regression.utils import convolve_mean
from covid_model_seiir_pipeline.forecasting.model import ODERunner

COL_TEMP = 'temperature'
COL_TESTING = 'testing_reference'
COL_POP_DENSITY = 'proportion_over_1k'
COL_MOBILITY = 'mobility_lift'


class ModelRunner:
    def __init__(self):
        self.ode_model = None

    def fit_beta_ode(self, ode_process_input):
        self.ode_model = ODEProcess(ode_process_input)
        self.ode_model.process()

    def get_beta_ode_fit(self, path=None):
        if self.ode_model is None:
            assert path is not None, 'Must fit_beta_ode or provide the path ' \
                                     'to the fit result.'
            return pd.read_csv(path)
        else:
            return self.ode_model.create_result_df()

    def get_beta_ode_params(self, path=None):
        if self.ode_model is None:
            assert path is not None, 'Must fit_beta_ode or provide the path ' \
                                     'to the fit parameters.'
            return pd.read_csv(path)
        else:
            return self.ode_model.create_params_df()

    def get_beta_start_end_dates(self, path=None):
        if self.ode_model is None:
            assert path is not None, 'Must fit_beta_ode or provide the path ' \
                                     'to the dates.'
            return pd.read_csv(path)
        else:
            return self.ode_model.create_start_end_date_df()

    def save_beta_ode_result(self, fit_file, params_file):
        """Save result from beta ode fit.

        Args:
            fit_file (str): fit file path to save to
            params_file (str): params file to save to
        """
        assert self.ode_model is not None, 'Must fit_beta_ode first.'
        # save ode fit
        self.get_beta_ode_fit().to_csv(fit_file, index=False)
        # save other parameters
        self.get_beta_ode_params().to_csv(params_file, index=False)

    def predict_beta_forward(self, covmodel_set, df_cov, df_cov_coef, col_t, col_group, col_beta='ln_beta_pred'):
        regressor = BetaRegressor(covmodel_set)
        regressor.load_coef(df=df_cov_coef)
        return predict(regressor, df_cov, col_t, col_group, col_beta)

    @staticmethod
    def covmodels_prod():
        cov_temp = CovModel(col_cov=COL_TEMP, use_re=False, bounds=np.array([-np.inf, 0.0]))
        cov_testing = CovModel(col_cov=COL_TESTING, use_re=False, bounds=np.array([-np.inf, 0.0]))
        cov_pop_density = CovModel(col_cov=COL_POP_DENSITY, use_re=False, bounds=np.array([0.0, np.inf]))
        cov_mobility = CovModel(col_cov=COL_MOBILITY, use_re=True, bounds=np.array([0.0, np.inf]), re_var=np.inf)
        cov_intercept = CovModel(col_cov='intercept', use_re=True, re_var=np.inf)
        return cov_temp, cov_testing, cov_pop_density, cov_mobility, cov_intercept

    def fit_beta_regression_prod(self, ordered_covmodel_sets, mr_data, path,
                                 df_cov_coef=None, std=1.0, sequential=False):
        # If priors on coefficients are assigned from a data frame, add them
        if df_cov_coef is not None:
            coef_values = {col_name: df_cov_coef[col_name].to_numpy().mean() for col_name in df_cov_coef.columns}
            for covmodelset in ordered_covmodel_sets:
                for covmodel in covmodelset.cov_models:
                    if (covmodel.col_cov in coef_values) and not covmodel.use_re:
                        if covmodel.gprior is not None:
                            covmodel.gprior[0] = coef_values[covmodel.col_cov]
                        else:
                            covmodel.gprior = np.array([coef_values[covmodel.col_cov], std])
        # The regressor type depends on whether the regression is sequential
        if sequential:
            regressor = BetaRegressorSequential(ordered_covmodel_sets, default_std=std)
        else:
            covmodels = []
            for covmodel_set in ordered_covmodel_sets:
                covmodels.extend(covmodel_set.cov_models)
            covmodels_set_comb = CovModelSet(covmodels)
            regressor = BetaRegressor(covmodels_set_comb)
        regressor.fit(mr_data)
        print(regressor.cov_coef)
        regressor.save_coef(path)

    def predict_beta_forward_prod(self, covmodel_set, df_cov, df_cov_coef,
                                  col_t, col_group, avg_window=0):
        df = self.predict_beta_forward(covmodel_set, df_cov, df_cov_coef, col_t, col_group, 'ln_beta_pred')
        beta_pred = np.exp(df['ln_beta_pred']).values[None, :]
        beta_pred = convolve_mean(beta_pred, radius=[0, avg_window])
        df['beta_pred'] = beta_pred.ravel()
        return df

    @staticmethod
    def forecast(model_specs, init_cond, times, betas, thetas=None, dt=0.1):
        """
        Solves ode for given time and beta

        Arguments:
            model_specs (SeiirModelSpecs): specification for the model. See
                covid_model_seiir_pipeline.forecasting.model.SeiirModelSpecs
                for more details.
                example:
                    model_specs = SeiirModelSpecs(
                        alpha=0.9,
                        sigma=1.0,
                        gamma1=0.3,
                        gamma2=0.4,
                        N=100,  # <- total population size
                    )

            init_cond (np.array): vector with five numbers for the initial conditions
                The order should be exactly this: [S E I1 I2 R].
                example:
                    init_cond = [96, 0, 2, 2, 0]

            times (np.array): array with times to predict for
            betas (np.array): array with betas to predict for
            thetas (np.array): optional array with a term indicating size of SEIIR
                adjustment by day. If None, defaults to an adjustment of 0. If
                not None, must have the same dimensions as betas.
            dt (float): Optional, step of the solver. I left it sticking outside
                in case it works slow, so you can decrease it from the IHME pipeline.

        Returns:
            result (DataFrame):  a dataframe with columns ["S", "E", "I1", "I2", "R", "t", "beta"]
            where t and beta are times and beta which were provided, and others are solution
            of the ODE
        """
        forecaster = ODERunner(model_specs, init_cond, dt=dt)
        return forecaster.get_solution(times, beta=betas, theta=thetas)
