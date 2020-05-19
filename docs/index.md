# COVID19 SEIIR Modeling Pipeline at IHME

This page documents the usage of the COVID-19 SEIIR Modeling Pipeline used
at IHME.

## Getting Started

Here we will walk through installing the pipeline in an environment on the IHME
cluster and running the pipeline.

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

The directory that is used for infectionator inputs is `/ihme/covid-19/seir-inputs` and the directory
that is used for covariate inputs is `/ihme/covid-19/seir-covariates`.

The main directories for outputs within `/ihme/covid-19/seir-pipeline-outputs` are:

- `ode`: ode versions and outputs
- `regression`: regression versions and outputs
- `forecast`: forecast versions and outputs
- `covariate`: covariate cache versions for both forecasts and regressions
- `metadata-inputs`: location metadata snapshots from the hierarchies with some
    modifications explained in the README.md files

Within an ODE version, the directory structure is:
- `betas`: location-specific folders with draw files for each
    component fit -- also includes predicted beta based on the regression
- `parameters`: draw-specific files with simulated alpha, sigma, gamma1, and gamma2
    parameters
- `locations.csv`: a snapshot of the location IDs that was used in this run
- `settings.json`: the settings for the ODE version

Within a regression version, the directory structure is:

- `coefficients`: draw-specific files with coefficients from the regression
    fit. This is what is read in when a new regression version uses a prior
    coefficient version.
- `diagnostics`: regression diagnostic plots
- `settings.json`: the settings for the regression version

Within a forecast version, the directory structure is:

- `beta_scaling`: a csv of the scaling factor that was produced for each location
    and draws are within the files -- scaling factor makes past betas line up 
    with future betas
- `component_draws`: folders by location with draw-specific files of the
    SEIIR components
- `output_draws`: files by location and type of draw which are the final
    outputs. includes cases, deaths, and r effective. This is post-splicing.
- `diagnostics`: plots of SEIIR component draws, final draws, and beta residuals
- `settings.json`: settings for the forecast version

### Versions

To create an ODE, regression, or forecast version, you pass arguments to `ODEVersion`,
`RegressionVersion`, or `ForecastVersion`. Since creating a version requires
caching locations, covariates and creating the directory structure, we suggest using
the utilities functions `create_ode_version`, `create_regression_version`, 
`create_forecast_version`, or
if you are doing a run creating a *new* ODE, regression, and forecast version that you want to have
the same name, `create_run`.

The versions build on each other. An ODE version can be created by itself, but
you must have an existing ODE version that you pass to a regression version,
and you must have an existing regression version to pass to a forecast version.

The arguments to `create_ode_version` are:

- `version_name (str)`: (*) the name of the ODE version
- `infection_version (str)`: (*) the infectionator version to use
- `location_set_version_id (int)`: (*) the location metadata version to use (MUST BE IN `metadata-inputs`!)
- `n_draws (int)`: (*) number of draws to run
- `degree (int)`: (*) degree of the spline to fit on new infections to get beta
- `knots (int)`: (*) knot positions for the spline
- `day_shift (Tuple[int])`: (*) Will use today + `day_shift` - lag 's data in the beta regression but sample
    the day shift from the range passed in
- `alpha (Tuple[float])`: (*) a 2-length tuple that represents the range of alpha values to sample
- `sigma (Tuple[float])`: (*) a 2-length tuple that represents the range of sigma values to sample
- `gamma1 (Tuple[float])`: (*) a 2-length tuple that represents the range of gamma1 values to sample
- `gamma2 (Tuple[float])`: (*) a 2-length tuple that represents the range of gamma2 values to sample
- `solver_dt (float)`: (*) step size for the ODE solver

The arguments to `create_regression_version` are:

- `version_name (str)`: the name of the regression version
- `ode_version (str)`: the ode version to use
- `covariate_version (str)`: (*) covariate version to use
- `covariate_draw_dict (Dict[str, bool])`: (*) a dictionary of covariate name to whether or not the covariate has draws
- `coefficient_version (str)`: (*) the regression version of coefficient estimates to use
- `covariates (Dict[str: Dict])`: (*) elements of the inner dict:
    - `"use_re" (bool)`: use random effects or not
    - `"gprior" (np.array)`: mean and standard deviation for the Gaussian prior on the fixed effect
    - `"bounds" (np.array)`: lower and upper bounds for the sum of fixed + random effects
    - `"re_var": (float)`: the variance of the random effect
- `covariates_order (List[List[str]])`: (*) list of lists of covariate names that will be
        sequentially added to the regression

The arguments to `create_forecast_version` are:

- `version_name (str)`:  the name of the forecast version
- `regression_version (str)`: the regression version to use
- `covariate_version (str)`: (*) covariate version to use
- `covariate_draw_dict (Dict[str, bool])`: (*) a dictionary of covariate name to whether or not the covariate has draws.
    This is separate from the regression version `covariate_draw_dict` because you may want to use draws for the
    forecasts but not in the regression fitting. This allows you to do that.

**NOTE**: The `covariate_draw_dict` has not been tested with `True` for any of the covariates because the input
data does not exist yet to test.

To create a run that will have an ode, regression, and forecast version (for example, the baseline scenario),
use the `create_run` utility function which takes all the unique arguments above that are starred with (*). An example
of using the `create_run` function is below:

```python
from seiir_model_pipeline.core.utils import create_run

covariates = dict(
    temperature=dict(use_re=False, gprior=[0.0, 1000], bounds=[-1000., 0.], re_var=1.),
    mobility_lift=dict(use_re=True, gprior=[0.0, 1e-1], bounds=[0., 1000.], re_var=1.),
    proportion_over_1k=dict(use_re=False, gprior=[0.0, 10000], bounds=[0., 1000.], re_var=1.),
    testing_reference=dict(use_re=False, gprior=[0.0, 1000], bounds=[-1000., 0.], re_var=1.)
)
covariate_draw_dict = {
    'mobility_lift': False,
    'proportion_over_1k': False,
    'temperature': False,
    'testing_reference': False
}
covariate_orders = [['mobility_lift'], ['proportion_over_1k'], ['temperature'], ['testing_reference']]
create_run(
    infection_version='2020_05_09.01',
    covariate_version='2020_05_09.01',
    n_draws=10, location_set_version_id=0,
    degree=3, day_shift=5, knots=[0.0, 0.25, 0.5, 0.75, 1.0],
    covariates=covariates,
    alpha=[0.9, 1], sigma=[1./5., 1./3.], gamma2=[1./3., 1],
    version_name='develop/refactor18',
    covariate_draw_dict=covariate_draw_dict, covariates_order=covariate_orders
)
```

### Launching a Pipeline Run

To launch a pipeline run, you must first create an ODE version and/or a regression version 
and/or a forecast version.

The `versioner` module has a helper function `create_run` that creates all three
 with the same name for a full run, and with all the keyword arguments
that you would pass to `ODEVersion`, `RegressionVersion`, and `ForecastVersion` (see [versions](#versions)).

Once you have a version, open up a screen and qlogin on the dq on the cluster with at least three
threads, 5G of memory, and a few hours of runtime (the pipeline is quick, but the diagnostics
take a long time). The `run` script builds the Jobmon workflow and monitors the tasks
until the whole pipeline is complete:

```
run --ode-version {ODE_VERSION} --regression-version {REG_VERSION} --forecast-version {FOR_VERSION} --run-splicer --create-diagnostics
```

If you don't want to create diagnostics, remove the `--create-diagnostics` flag. Likewise, if you don't
want to run the splicer to create infections and deaths from the ODE compartments, remove the
`--run-splicer` flag.

If you only want to run one of the three parts, just include that one part (or two).
For example, if you *only* want to run a forecast version off of a previous regression version, 
remove the `--regression-version` flag after you have created a forecast version
tagging a specific regression version. Similarly, if you *only* want to run an ODE
version, then don't pass a regression or forecast version.

You can view the status of the workflow and individual tasks in the workflow by viewing
the Jobmon database. Instructions for accessing the permanent Jobmon database are
[here](https://hub.ihme.washington.edu/display/DataScience/Jobmon+Release+Notes) under the
10/7/2019 release notes.

## Core Components

The main scripts used in the pipeline are stored in `executors`. They represent sections
of the pipeline and are briefly described below.

#### Run

The `run` script is the main script that [launches a full pipeline run](#launching-a-pipeline-run).
It needs to know a regression version, a forecast version, and whether or not to run the splicer and
to create diagnostics. It uses Jobmon to create a workflow with `n_draws` number
of [regression tasks](#beta-regression), then [forecasting tasks](#beta-forecast) for each location,
then [splicing](#splicing) tasks for each forecast task, and finally one [diagnostics](#diagnostics)
task.

#### ODE Fit

The ODE fit task reads infection data for one draw across all locations.
The tasks are parallelized across draw because each draw is independent.

First, it fits a spline using the [`xspline` package](https://github.com/zhengp0/xspline)
to the daily infection data provided from the *Infectionator* over time. It
 then solves
an SEIIR ODE using the spline fit to get estimates for each of the SEIIR
 compartments. At last it obtains the beta from the spline
 and all the compartments of the SEIIR ODE.

#### Beta Regression

The beta regression task loads the outputs of an ODE fit and covariate data based
on the settings provided when you created a version

It fits a regression using covariates provided in your specs
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
to get estimates for each component of the SEIIR. The ODE uses the initial
 conditions saved
from the past SEIIR components from the [regression task](#beta-regression).

#### Splicing

The goal of the splicer is to create infections and deaths from the SEIIR compartments.
For each draw, it takes the output draws for the past and the future for new infections
and concats them together, while creating newE, which are the equivalent of "cases" from the infectionator, which
is calculated as the difference in susceptible numbers across time points.
Then it fills in all dates until `today` with the infectionator outputs rather than the modeled results that we have.

After splicing cases, it takes the new cases data from the splicing above and multiplies the cases by
the infection fataility ratio (IFR) for this specific draw, and takes into account the draw-specific lag
also from the infectionator, to create deaths draws. Those death draws are then replaced with infectionator
death draws up until `today`. Those are then replaced with infectionator **mean** deaths based on the `obs_deaths`
indicator column available in the infectionator outputs. This fills in the smooth death data **before** modeling.

Finally, it creates R effective based on the compartments. It saves these three outputs in different files.

#### Diagnostics

Diagnostics are created for forecast versions (`create_forecast_diagnostics`),
regression versions (`create_regression_diagnostics`) and what happens in between
which is the scaling of the beta for the forecast to match the regression
(`create_scaling_diagnostics`). The diagnostics are created using functions within
the `seiir_model_pipeline.diagnostics` directory. To see their directories,
see [directories](#directories).
