Model calibration
=================

The importance of having coherent estimates of population immunity is 
essential to forecasting COVID-19 transmission at this stage in the 
pandemic. Many key drivers of transmission early in the pandemic, such as 
NPIs, are now sparsely implemented, leaving population attributes like 
past infection an vaccination rates, in addition to parameters such as
breakthrough potential and waning immunity, as key determinants for 
forecasting transmission of these highly infectious variants.

The lower severity of the Omicron lineage, coupled with decline in quality 
of reporting for epidemiological surveillance, complicate the task of 
estimating historical infections based on reported cases, hospital 
admissions, and deaths, requiring a more contextually sensitive approach. 
We achieve this by deriving location- and variant-specific risk-ratios 
(IDR, IHR, and IFR) based on the relative difference in infections 
estimated by a naive model using only global risk-ratios compared to those 
estimated by a counterfactual model of transmission that controls for known
drivers.

The steps for running this system are described here. These steps 
are run for each variant.

1) Run naive model
--------------
First, create a new folder with copies of the most up-to-date kappa scaling 
factors (found here: `src/covid_model_seiir_pipeline/pipeline/fit/model/kappa_scaling_factors/`), 
but excluding the yaml file for the variant you are optimizing.

Replace the `calibration_type` argument in the fit specification with your 
new version. Then, add the variant you are optimizing to the `sero_exclude_variants` 
argument in that spec file.

Run the fit stage of the model.

2) Generate counterfactual inputs and run counterfactual
--------------------------------------------------------
Add your fit version and the variant you are optimizing to 
`src/covid_model_seiir_pipeline/side_analysis/calibration/scenario_generation.ipynb` 
and run `build_output_scenario_version` (probably fine to use most recent 
prod regression).

Add the counterfactual inputs and fit versions to a counterfactual spec file 
and run the counterfactual.

3) Generate local kappa scaling factors
---------------------------------------
Use the counterfactual version you've created as the `reference_version`, 
the fit version as the `comparator_version`, and the variant you are optimizing 
(leaving `RUN = 'LOCAL'`) in `src/covid_model_seiir_pipeline/side_analysis/calibration/2022_08_12_optimize_infections.ipynb`
and run (`compile_inputs` -> `calculate_kappa_scalars` -> `store_kappa_scalars`). 
This will spit out a yaml file with the name of your variant in 
`src/covid_model_seiir_pipeline/pipeline/fit/model/kappa_scaling_factors/` 
that contains your new kappa scaling factors. Move this into the folder you 
created in step 1.

4) Run optimized model
----------------------
Remove your variant from `sero_exclude_variants` in the fit spec file and 
re-run the fit stage (and continue through to forecasts).
