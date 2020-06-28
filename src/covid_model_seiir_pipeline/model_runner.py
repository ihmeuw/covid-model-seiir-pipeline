from covid_model_seiir_pipeline.forecasting.model import ODERunner


class ModelRunner:

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
