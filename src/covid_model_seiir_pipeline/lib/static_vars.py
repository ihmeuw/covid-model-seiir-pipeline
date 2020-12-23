
REGRESSION_SPECIFICATION_FILE = 'regression_specification.yaml'
FORECAST_SPECIFICATION_FILE = 'forecast_specification.yaml'
POSTPROCESSING_SPECIFICATION_FILE = 'postprocessing_specification.yaml'

SEIIR_COMPARTMENTS = ['S', 'E', 'I1', 'I2', 'R']

VACCINE_SEIIR_COMPARTMENTS = [
    'S',   'E',   'I1',   'I2',   'R',    # Unvaccinated
    'S_u', 'E_u', 'I1_u', 'I2_u', 'R_u',  # Vaccinated and unprotected
    'S_p', 'E_p', 'I1_p', 'I2_p', 'R_p',  # Vaccinated and protected
    'M',                                  # Vaccinated and immune
]

COL_BETA = 'beta'
COL_GROUP = 'loc_id'
COL_DATE = 'date'
COL_INTERCEPT = 'intercept'

DAYS_PER_WEEK = 7

# Columns from infectionator inputs
INFECTION_COL_DICT = {
    'COL_DATE': 'date',
    'COL_CASES': 'cases_draw',
    'COL_POP': 'pop',
    'COL_LOC_ID': 'location_id',
    'COL_DEATHS': 'deaths_draw',
    'COL_ID_LAG': 'i_d_lag',
    'COL_OBS_DEATHS': 'obs_deaths',
    'COL_OBS_CASES': 'obs_infecs',
    'COL_DEATHS_DATA': 'deaths_mean'
}

# Columns from covariates inputs
COVARIATE_COL_DICT = {
    'COL_DATE': 'date',
    'COL_OBSERVED': 'observed',
    'COL_LOC_ID': 'location_id'
}

# The key for the observed column
OBSERVED_DICT = {
    'observed': 1.,
    'forecasted': 0.
}
