import numpy as np
from typing import List, Iterable
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.interpolate import UnivariateSpline


class SplineModel(BaseEstimator, RegressorMixin):
    def __init__(self, s=None, k=3):
        self.s = s
        self.k = k
        self.spline = None

    def fit(self, x, y):
        self.spline = UnivariateSpline(x, y, k=self.k, s=self.s)
        return self

    def predict(self, x):
        assert self.spline is not None, "Fit spline model first"
        return self.spline(x)


def get_stds(x: List, y: List, ss: Iterable[float] = None, cv=20, mode="fast"):
    """
    Recovers standard errors by fitting smoothing splines

    :param x: (List) x_axis data
    :param y: (List) y_axis data
    :param ss: (List of floats: smoothing parameters for grid search. If None then uses and
        a pre-defined extensive set from 10 to 1e6.
    :param cv: number of CV draws
    :param mode: (str) which method to use. Three options:
        - "super-fast": fits one spline with grid-search cv and outputs
            the absolute residuals as stds.
            The fastest but the least correct method from statistical point of view.
        - "fast": finds the optimal smoothing parameter via grid search and then
            for each point it fits a spline for all the data except that point and outputs
            the absolute residual as stds. Not completely correct since the std estimation
            for a point is still affected by that point through the smoothing parameter,
            but in practice works as good as the "correct" method.
        - "correct": fits a separate spline (with CV cross-validation) for each point.
            The most correct but the absolutely slowest method.

    :return: stds (np.array) -- estimations of standard errors
    """
    assert len(x) == len(y), "x and y should have the same length"
    if ss is None:
        ss = ([i for i in range(10, 100, 10)]
              + [i for i in range(100, 1000, 100)]
              + [i for i in range(1000, 10000, 1000)]
              + [i for i in range(10000, 100000, 10000)]
              + [i for i in range(int(1e5), int(1e6), int(1e5))])

    num_points = len(x)
    stds = np.zeros(num_points)
    param_grid = {
        "k": [3],  # this preserves the spline to be cubic
        "s": ss
    }

    np.random.seed(42)  # fixes CV splits for reproducibility
    if mode == "super-fast":
        model = SplineModel()
        grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=cv)
        grid_search.fit(x, y)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(x)
        return np.abs(np.array(y) - y_pred)

    elif mode == "fast":
        model = SplineModel()
        grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=cv)
        grid_search.fit(x, y)
        best_s = grid_search.best_estimator_.s
        for i in range(num_points):
            xi = x.pop(i)
            yi = y.pop(i)
            model = SplineModel(s=best_s)
            model.fit(x, y)
            yi_pred = model.predict(xi)
            stds[i] = abs(yi - yi_pred)
            x.insert(i, xi)
            y.insert(i, yi)
        return stds

    elif mode == "correct":
        for i in range(num_points):
            xi = x.pop(i)
            yi = y.pop(i)
            model = SplineModel()
            np.random.seed(i)  # fixes CV splits for reproducibility
            grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=cv)
            grid_search.fit(x, y)
            best_model = grid_search.best_estimator_
            yi_pred = best_model.predict(xi)
            stds[i] = abs(yi - yi_pred)
            x.insert(i, xi)
            y.insert(i, yi)
        return stds
    else:
        raise ValueError("Unknown mode: %s" % mode)


def estimate_standard_error(window_size, **kwargs):
    """
    Estimates the standard error for each data point based on estimated residuals from the
    `get_stds` function that uses smoothing splines. Computes the standard deviation of the residuals
    over a window size. Arrays must therefore be *ordered* in some meaningful way -- e.g. chronologically.

    :param window_size: the window over which to compute the standard deviation of residuals
        from the smoothing spline
    :param kwargs: keyword arguments to the `get_stds` function
    :return: 1D np.array of size `y` and `x`, the input arrays
    """
    stds = get_stds(**kwargs)
    return pd.Series(stds).rolling(
        window=window_size, center=True, min_periods=1
    ).apply(lambda x: np.std(x)).values
