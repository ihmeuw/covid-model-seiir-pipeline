"""Port of https://github.com/ihmeuw-msca/SLIME

Doing a direct include here so I can figure out how to speed up the regression.
"""
from typing import Dict, Any

import numpy as np
from scipy.linalg import block_diag
import scipy.optimize as sciopt


class MRData:

    def __init__(self, df, col_group, col_obs, col_obs_se=None, col_covs=None):
        self.df_original = df.copy()
        self.col_group = col_group
        self.col_obs = col_obs
        self.col_obs_se = col_obs_se
        self.col_covs = [] if col_covs is None else col_covs

        # add intercept as default covariates
        df['intercept'] = 1.0
        if 'intercept' not in self.col_covs:
            self.col_covs.append('intercept')

        # add observation standard error
        if self.col_obs_se is None:
            self.col_obs_se = 'obs_se'
            df[self.col_obs_se] = 1.0

        assert self.col_group in df
        assert self.col_obs in df
        assert self.col_obs_se in df
        assert all([name in df for name in self.col_covs])
        self.df = df[[self.col_group, self.col_obs, self.col_obs_se] +
                     self.col_covs].copy()

        # Use merge sort for stability.
        self.df.sort_values(col_group, inplace=True, kind='mergesort')
        self.groups, self.group_sizes = np.unique(self.df[self.col_group],
                                                  return_counts=True)

        self.group_idx = sizes_to_indices(self.group_sizes)
        self.num_groups = len(self.groups)
        self.num_obs = self.df.shape[0]

    def df_by_group(self, group):
        """Divide data by group.
        Args:
            group (any): Group id in the data frame.
        Returns:
            pd.DataFrame: The corresponding data frame.
        """
        assert group in self.groups
        return self.df[self.df[self.col_group] == group]


class MRModel:
    """Linear MetaRegression Model.
    """

    def __init__(self, data: MRData, cov_models: 'CovModelSet'):
        """Constructor of the MetaRegression Model.

        Args:
            data (MRData): Data object
            cov_models (CovModelSet): Covariate models.
        """
        self.data = data
        self.cov_models = cov_models

        # unpack data
        self.obs = data.df[data.col_obs].values
        self.obs_se = data.df[data.col_obs_se].values

        # pass data into the covariate models
        self.cov_models.attach_data(self.data)
        self.bounds = self.cov_models.extract_bounds()
        self.opt_result = None
        self.result = None

    def objective(self, x):
        """Objective function for the optimization.

        Args:
            x (np.ndarray): optimization variable.
        """
        # data
        prediction = self.cov_models.predict(x)
        val = 0.5*np.sum(((self.obs - prediction)/self.obs_se)**2)

        # prior
        val += self.cov_models.prior_objective(x)

        return val

    def gradient(self, x):
        """Gradient function for the optimization.

        Args:
            x (np.ndarray): optimization variable.
        """
        prediction = self.cov_models.predict(x)
        residual = self.obs - prediction
        return self.cov_models.gradient(x, residual, self.obs_se)

    def hessian(self, x: np.array) -> np.ndarray:
        """Hessian function for the optimization.

        Args:
            x (np.array): optimization variable.

        Returns:
            np.ndarray: Hessian matrix.
        """
        return self.cov_models.hessian(x, self.obs_se)

    def fit_model(self, x0=None, options=None):
        """Fit the model, including initial condition and parameter.
        Args:
            x0 (np.ndarray, optional):
                Initial guess for the optimization variable.
            options (None | dict):
                Optimization solver options.
        """
        if x0 is None:
            x0 = np.zeros(self.cov_models.var_size)
        self.opt_result = sciopt.minimize(
            fun=self.objective,
            x0=x0,
            jac=self.gradient,
            method='L-BFGS-B',
            bounds=self.bounds,
            options=options
        )

        self.result = self.cov_models.process_result(self.opt_result.x)

    def sample_soln(self, num_draws: int = 1) -> Dict[Any, np.ndarray]:
        """Create draws for the solution.

        Args:
            num_draws (int, optional): Number of draws. Defaults to 1.

        Returns:
            Dict[Any, np.ndarray]:
                Dictionary with group_id as the key solution draws as the value.
        """
        if self.opt_result is None or self.result is None:
            RuntimeError('Fit the model first before sample the solution.')

        hessian = self.hessian(self.opt_result.x)
        info_mat = np.linalg.inv(hessian)

        samples = np.random.multivariate_normal(mean=self.opt_result.x,
                                                cov=info_mat,
                                                size=num_draws)
        _soln_samples = [
            self.cov_models.process_result(
                np.minimum(np.maximum(
                    samples[i], self.bounds[:, 0]), self.bounds[:, 1])
            )
            for i in range(num_draws)
        ]
        soln_samples = {
            g: np.vstack([
                _soln_samples[i][g]
                for i in range(num_draws)
            ])
            for g in self.cov_models.groups
        }
        return soln_samples


class CovModel:
    """Single covariate model.
    """

    def __init__(self, col_cov,
                 use_re=False,
                 bounds=(-np.inf, np.inf),
                 gprior=(0.0, np.inf),
                 re_var=1.0):
        """Constructor CovModel.

        Args:
            col_cov (str): Column for the covariate.
            use_re (bool, optional): If use the random effects.
            bounds (np.ndarray | None, optional):
                Bounds for the covariate multiplier.
            gprior (np.ndarray | None, optional):
                Gaussian prior for the covariate multiplier.
            re_var (float, optional):
                Variance of the random effect, if use random effect.
        """
        self.col_cov = col_cov
        self.use_re = use_re
        self.bounds = np.array(bounds)
        self.gprior = np.array(gprior)
        self.re_var = re_var

        self.name = self.col_cov
        self.var_size = None

        self.cov = None
        self.cov_mat = None
        self.cov_scale = None

        self.group_idx = None
        self.group_sizes = None

    def attach_data(self, data: MRData):
        """Attach the data.

        Args:
            data (MRData): MRData object.
        """
        self.group_idx = data.group_idx
        self.group_sizes = data.group_sizes
        assert self.col_cov in data.df
        if self.use_re:
            self.var_size = data.num_groups
        else:
            self.var_size = 1

        cov = data.df[self.col_cov].values
        cov_scale = np.linalg.norm(cov)
        assert cov_scale > 0.0
        self.cov = cov/cov_scale
        self.cov_scale = cov_scale
        if self.use_re:
            self.cov_mat = block_diag(*[
                self.cov[self.group_idx[i]][:, None]
                for i in range(data.num_groups)
            ])
        else:
            self.cov_mat = self.cov[:, None]

    def detach_data(self):
        """Detach the object from the data.
        """
        self.var_size = None
        self.cov = None
        self.cov_mat = None
        self.cov_scale = None
        self.group_sizes = None
        self.group_idx = None

    def get_cov_multiplier(self, x):
        """Transform the effect to the optimization variable.

        Args:
            x (np.ndarray): optimization variable.
        """
        if self.use_re:
            return np.repeat(x, self.group_sizes)
        else:
            return np.repeat(x, self.group_sizes.sum())

    def predict(self, x):
        """Predict for the optimization problem.

        Args:
            x (np.ndarray): optimization variable.
        """
        return self.cov*self.get_cov_multiplier(x)

    def prior_objective(self, x):
        """Objective related to prior.

        Args:
            x (np.ndarray): optimization variable.
        """
        val = 0.0

        x = x/self.cov_scale
        # random effects priors
        if self.use_re and np.isfinite(self.re_var):
            val += 0.5*np.sum((x - np.mean(x))**2)/self.re_var

        # Gaussian prior for the effects
        if np.isfinite(self.gprior[1]):
            val += 0.5*np.sum(((x - self.gprior[0])/self.gprior[1])**2)

        return val

    def gradient(self, x, residual, obs_se):
        """Compute the gradient for the covariate multiplier.

        Args:
            x (np.ndarray): optimization variable.
            residual (np.ndarray): residual array, observation minus prediction.
            obs_se (np.ndarray): observation standard deviation.

        Return:
            np.ndarray: gradient
        """
        grad = -(self.cov_mat.T/obs_se).dot(residual/obs_se)
        x = x/self.cov_scale
        if self.use_re and np.isfinite(self.re_var):
            grad += ((x - np.mean(x))/self.re_var)/self.cov_scale

        if np.isfinite(self.gprior[1]):
            grad += ((x - self.gprior[0])/self.gprior[1]**2)/self.cov_scale

        return grad

    def extract_bounds(self):
        """Extract the bounds for the optimization problem.
        """
        bounds = self.bounds*self.cov_scale
        return np.repeat(bounds[None, :], self.var_size, axis=0)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.col_cov})'

    def __eq__(self, other: 'CovModel') -> bool:
         return (
            np.all(self.bounds == other.bounds)
            and self.col_cov == other.col_cov
            and np.all(self.cov == other.cov)
            and np.all(self.cov_mat == other.cov_mat)
            and self.cov_scale == other.cov_scale
            and np.all(self.gprior == other.gprior)
            and np.all(np.hstack(self.group_idx) == np.hstack(other.group_idx))
            and np.all(self.group_sizes == other.group_sizes)
            and self.re_var == other.re_var
            and self.use_re == other.use_re
            and self.var_size == other.var_size
        )

    @classmethod
    def from_specification(cls, covariate):
        return cls(
            col_cov=covariate.name,
            use_re=covariate.use_re,
            bounds=np.array(covariate.bounds),
            gprior=np.array(covariate.gprior),
            re_var=covariate.re_var,
        )


class CovModelSet:
    """A set of CovModel.
    """

    def __init__(self, cov_models, data=None):
        """Constructor of the covariate model set.

        Args:
            cov_models (list{CovModel}): A list of covariate set.
            data (MRData | None, optional): Data to be attached.
        """
        assert isinstance(cov_models, list)
        assert all([isinstance(cov_model, CovModel)
                    for cov_model in cov_models])
        self.cov_models = cov_models
        self.num_covs = len(self.cov_models)

        self.var_size = None
        self.var_sizes = None
        self.var_idx = None
        self.groups = None
        self.num_groups = None

        if data is not None:
            self.attach_data(data)

    def attach_data(self, data):
        """Attach the data.

        Args:
            data (MRData): MRData object.
        """
        for cov_model in self.cov_models:
            cov_model.attach_data(data)

        self.var_sizes = np.array([
            cov_model.var_size for cov_model in self.cov_models
        ])
        self.var_size = np.sum(self.var_sizes)
        self.var_idx = sizes_to_indices(self.var_sizes)
        self.groups = data.groups
        self.num_groups = data.num_groups

    def detach_data(self):
        """Detach the object from the data.
        """
        for cov_model in self.cov_models:
            cov_model.detach_data()

        self.var_size = None
        self.var_sizes = None
        self.var_idx = None

        self.groups = None
        self.num_groups = None

    def predict(self, x):
        """Predict for the optimization.

        Args:
            x (np.ndarray): optimization variable.
        """
        return np.sum([
            cov_model.predict(x[self.var_idx[i]])
            for i, cov_model in enumerate(self.cov_models)
        ], axis=0)

    def prior_objective(self, x):
        """Objective related to prior.

        Args:
            x (np.ndarray): optimization variable.
        """
        return np.sum([
            cov_model.prior_objective(x[self.var_idx[i]])
            for i, cov_model in enumerate(self.cov_models)
        ])

    def gradient(self, x, residual, obs_se):
        """Gradient function.

        Args:
            x (np.ndarray): optimization variable.
            residual (np.ndarray): residual array, observation minus prediction.
            obs_se (np.ndarray): observation standard deviation.

        Return:
            np.ndarray: gradient
        """
        return np.hstack([
            cov_model.gradient(x[self.var_idx[i]], residual, obs_se)
            for i, cov_model in enumerate(self.cov_models)
        ])

    def hessian(self, _: np.array, obs_se: np.array) -> np.ndarray:
        """Hessian function.

        Args:
            _ (np.array): optimization variable.
            obs_se (np.array): observation standard error.

        Returns:
            np.ndarray: Hessian matrix.
        """
        cov_mat = np.hstack([
            cov_model.cov_mat
            for cov_model in self.cov_models
        ])
        prior_diag = np.hstack([
            np.repeat(1.0/cov_model.gprior[1]**2, cov_model.var_size)
            for cov_model in self.cov_models
        ])
        return (cov_mat.T/obs_se**2).dot(cov_mat) + np.diag(prior_diag)

    def extract_bounds(self):
        """Extract the bounds for the optimization problem.
        """
        return np.vstack([
            cov_model.extract_bounds()
            for cov_model in self.cov_models
        ])

    def process_result(self, x):
        """Process the result, organize it by group and scale by the
        cov_scale.

        Args:
            x (np.ndarray): optimization variable.
        """
        coefs = np.vstack([
            x[self.var_idx[i]]/cov_model.cov_scale if cov_model.use_re else
            np.repeat(x[self.var_idx[i]], self.num_groups)/cov_model.cov_scale
            for i, cov_model in enumerate(self.cov_models)
        ])
        return {
            g: coefs[:, i]
            for i, g in enumerate(self.groups)
        }


def sizes_to_indices(sizes):
    """Converting sizes to corresponding indices.
    Args:
        sizes (numpy.ndarray):
            An array consist of non-negative number.
    Returns:
        list{range}:
            List the indices.
    """
    u_id = np.cumsum(sizes)
    l_id = np.insert(u_id[:-1], 0, 0)

    return [
        np.arange(l, u) for l, u in zip(l_id, u_id)
    ]
