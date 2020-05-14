import numpy as np
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt

from seiir_model_pipeline.core.std_recovery import SplineModel, get_stds

# This example is how to use the std_recovery functions -- how to estimate standard errors
# for data points using smoothing splines.

# Set up simulation parameters
size = 100
np.random.seed(42)
x = np.arange(0, np.pi, np.pi / size)
is_outlier = np.random.rand(size) < 0.1
outlier_ampl = 15
y = 5 * np.sin(x) + np.random.randn(size) + outlier_ampl * is_outlier * np.random.randn(size)
x = x.tolist()
y = y.tolist()

# Create a spline model and a parameter grid to search over
model = SplineModel()
ss = np.arange(500, 10000, 500)
param_grid = {
    "k": [3],  # this preserves the spline to be cubic
    "s": ss
}

# Find the best fit, *and* use smoothing splines to estimate the residuals for each data point
# Uses three methods to recover the residuals -- "super-fast", "fast" and "correct".
# "correct" and "fast" have nearly equivalent performance but "correct" is very slow.
grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=10)
grid_search.fit(x, y)
best_model = grid_search.best_estimator_
full_fit = best_model.predict(x)

stds_super_fast = get_stds(x, y, mode="super-fast", ss=ss)
print("super fast is done")
stds_fast = get_stds(x, y, mode="fast", ss=ss)
print("fast is done")
stds_correct = get_stds(x, y, mode="correct", ss=ss)
print("correct is done")

# Plot the results
fig = plt.figure(figsize=(15, 6))
grid = plt.GridSpec(1, 2)
ax1 = fig.add_subplot(grid[0, 0])
ax2 = fig.add_subplot(grid[0, 1])

x = np.array(x)
ax1.plot(x, 5*np.sin(x), label="true model")
ax1.plot(x, full_fit, label="spline fit")
ax1.scatter(x, y, label="data")
ax2.scatter(x, stds_super_fast, label="super-fast")
ax2.scatter(x, stds_fast, label="fast")
ax2.scatter(x, stds_correct, label="correct")
ax2.scatter(x, np.ones(size) + outlier_ampl * is_outlier, label="True")
ax1.legend()
ax2.legend()
plt.savefig("demo_spline_std.jpg")

