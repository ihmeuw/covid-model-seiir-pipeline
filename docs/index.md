# COVID19 SEIIR Modeling Pipeline at IHME

This page documents the usage of the COVID-19 SEIIR Modeling Pipeline used
at IHME.

## Getting Started

Here we will walk through installing the pipeline in an environment on the IHME
cluster and running

### Installation

The engine(s) behind the computation in the SEIIR pipeline are
from public and private GitHub repositories. This repository
contains a Makefile that you can use to install all of the repositories.
The only internal IHME package that this 
repository relies on is [Jobmon](https://scicomp-docs.ihme.washington.edu/jobmon/current/).

Note that in order to install the private repositories from [IHME
Math Sciences organization on GitHub](https://github.com/ihmeuw-msca), you will need to ask for
permissions to those repositories. Please contact [Peng Zheng](mailto:zhengp@uw.edu)
or [Marlena Bannick](mailto:mnorwood@uw.edu) to grant read permissions.

To install the repository into your own conda environment, clone the repository:

```
git clone https://stash.ihme.washington.edu/scm/cvd19/covid-model-seir-ode-opt.git
cd covid-model-seir-ode-opt
make install_env ENV_NAME={your conda environment name}
```

### Directories

All input and output filepaths for the SEIIR pipeline are coded within the `versioner` module.
The two main inputs are infection data from the Infectionator and covariates that are used in the beta
regressions. If the paths for these inputs ever changes, you need to change the filepaths
at the top of the `versioner` module.

Output folders and file names are also determined by the `versioner` module, in particular the
`Directories` object. The `Directories` object needs to read in a regression version
and/or a forecast version, the two halves of the pipeline. For a full run, you need
both a regression and forecast version.

### Versions

### Scripts

The main scripts used in the pipeline are stored in `executors`. They are briefly described below.

#### Run

The `run` script is the main script that [launches a full pipeline run](#launching-a-pipeline-run).
It needs to know a regression version, a forecast version, and whether or not to run the splicer and
to create diagnostics. It uses Jobmon to create a workflow with `n_draws` number
of [regression tasks](#beta-regression), then [forecasting tasks](#beta-forecast) for each location,
then [splicing](#splicing) tasks for each forecast task, and finally one [diagnostics](#diagnostics)
task.

#### Beta Regression

The beta regression task reads infection data for one draw across all locations and covariate data based
on the settings provided when you created a version. The regression tasks are parallelized across
draw because each draw is independent.

First, it fits a spline using the [`xspline` package](https://github.com/zhengp0/xspline)
to the daily infection data provided from the *Infectionator* over time. It
 then solves
an SEIIR ODE using the spline fit to get estimates for each of the SEIIR
 compartments. At last it obtains the beta from the spline
 and all the compartments of the SEIIR ODE.

After getting a smoothed beta curve, it fits a regression using covariates provided in your specs
to the smooth beta. The exact model it's fitting is actually a number of different
regression models done in stages, fixing coefficient values at different stages of the regression.
Some of these regressions have random effects. To date when this regression needs to change, we change
it in the core SEIIR model (on GitHub), not in the wrapper. Depending on needs, we may expose more knobs
so that the user can control the structure of the regression model directly from the wrapper.

The regression task saves the fitted betas from the spline, the SEIIR compartment estimates from
the ODE, the estimated coefficients from the regression, and the fitted beta values from the regression
(although these are never used, they are only saved for diagnostic purposes).

#### Beta Forecast

The beta forecast task is independent by location and draw, however it is currently only parallelized
by location because each draw is very fast.

For each draw, it reads in saved regression coefficients and ODE estimates for compartments **in the past**
from a [regression task](#beta-regression). The forecasting task uses future
covariate values to predict out the beta into the future using the saved coefficients.
To avoid discontinuities in the past and future predicted betas, we align them together
at the only shared time point (present) using a multiplicative scaling for the predicted betas 
(e.g. if last beta in the past time series is 0.95 times the first beta in the future time series, we multiply
the entire future beta time series by 0.95).

Then it solves an ODE moving forward into the future using this predicted beta time series 
to get estimates for the sizes of each of the SEIIR compartments. The ODE uses the initial conditions saved
from the past SEIIR components from the [regression task](#beta-regression).

#### Splicing

The goal of the splicer is to create infections and deaths from the SEIIR compartments.

#### Diagnostics.

### Launching a Pipeline Run

To launch a pipeline run, you must first create a regression version and/or a forecast version.
The `versioner` module has a helper function `create_run` that creates both a regression
and forecast version with the same name for a full run, and with all the keyword arguments
that you would pass to `RegressionVersion` and `ForecastVersion` (see [versions](#versions)).

Once you have a version, open up a screen and qlogin on the dq on the cluster with at least three
threads, 5G of memory, and a few hours of runtime (the pipeline is quick, but the diagnostics
take a long time). The `run` script builds the Jobmon workflow and monitors the tasks
until the whole pipeline is complete:

```
run --regression-version {REG_VERSION} --forecast-version {FOR_VERSION} --run-splicer --create-diagnostics
```

If you don't want to create diagnostics, remove the `--create-diagnostics` flag. Likewise, if you don't
want to run the splicer to create infections and deaths from the ODE compartments, remove the
`--run-splicer` flag.

You can view the status of the workflow and individual tasks in the workflow by viewing
the Jobmon database. Instructions for accessing the permanent Jobmon database are
[here](https://hub.ihme.washington.edu/display/DataScience/Jobmon+Release+Notes) under the
10/7/2019 release notes.
