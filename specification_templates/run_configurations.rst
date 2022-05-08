Preprocessing
=============

The `data.n_oversample_draws` is used to determine the available pool for resampling.
Ideally this number would be zero. Currently set to 50, which is too many and probably allows
some locations that should fail to make it through.  You can use the report at the end of the
fit logs to see what percentage of draws are failing and use that to tune here.
Perhaps start at 5 or 10 in the prod cycle and upscale if some locations really just
won't fit from parameterization but are close to inclusion.


Fit
===

For normal production or FHS run
--------------------------------
- `fit_parameters.omega_invasion`: Must be set to `''`.  Setting the invasion date to far
  in the future should work as well, but doesn't because I didn't handle an edge case.
- `rates_parameters.omicron_severity_parameterization`:  Doesn't matter
- `fit_parameters.kappa_omega`: Doesn't matter


For forecast scenarios paper
----------------------------
(Obviously adjust these if you work out a better parameterization.)

No new variant
++++++++++++++
Same as production in the past

- `fit_parameters.omega_invasion`: Must be set to `''`.  Setting the invasion date to far
  in the future should work as well, but doesn't because I didn't handle an edge case.
- `rates_parameters.omicron_severity_parameterization`:  Doesn't matter
- `fit_parameters.kappa_omega`: Doesn't matter

Omicron like variant
++++++++++++++++++++
- `fit_parameters.omega_invasion`: '2022-06-01' or any date we want goes here. The current
  implementation uses the omicron pattern of invasion shifted forward in time.
- `rates_parameters.omicron_severity_parameterization`:  'omicron'
- `fit_parameters.kappa_omega`: [2.5, 3.5]

Delta like variant
++++++++++++++++++++
- `fit_parameters.omega_invasion`: '2022-06-01' or any date we want goes here. The current
  implementation uses the omicron pattern of invasion shifted forward in time.
- `rates_parameters.omicron_severity_parameterization`:  'delta'
- `fit_parameters.kappa_omega`: [1.6, 2.0]

Average variant
+++++++++++++++
- `fit_parameters.omega_invasion`: '2022-06-01' or any date we want goes here. The current
  implementation uses the omicron pattern of invasion shifted forward in time.
- `rates_parameters.omicron_severity_parameterization`:  'average'
- `fit_parameters.kappa_omega`: [2.0, 2.8]

Deltacron variant
+++++++++++++++++
- `fit_parameters.omega_invasion`: '2022-06-01' or any date we want goes here. The current
  implementation uses the omicron pattern of invasion shifted forward in time.
- `rates_parameters.omicron_severity_parameterization`:  'delta'
- `fit_parameters.kappa_omega`: [2.5, 3.5]


Forecast
========

Still need to infection weight for the beta residual averaging.  Probably worth doing an
experiment before/during prod regardless of whether you get the automated kappa scaling to
work.

For normal production
---------------------
Standard configuration for `reference`, `best_masks` and `booster` scenario.

TODO: use a location-specific blend of the `reference` mask use
covariate scenario (which holds mask use constant) and the `relaxed` scenario that
starts scaling down mask use to a lower level dependant on historical levels.  Run both and
put in front of Steve or someone and have them pick for every location.

For FHS
-------
Standard configuration for `reference` only.

For forecast paper
------------------
For each of the five versions of the omega variant:
- `reference` scenario is production reference with the `relaxed` mask covariate
- `best_masks` is the production `best_masks` covariate
- `booster` is production booster scenario with the `relaxed` mask covariate


Postprocessing
==============
Same as normal.  You can fiddle with the resampling quantiles to deal with some of the
IFR explosions if you're in a pinch (I set the upper quantile to .925 to deal with burkina
for the FHS deliverable)

Diagnostics
===========
Same as normal.