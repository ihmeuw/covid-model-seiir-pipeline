

# Where location metadata is stored
LOCATION_METADATA_FILE_PATTERN = 'location_metadata_{lsvid}.csv'
# This is a list of locations used for a particular run
LOCATION_CACHE_FILE = 'locations.csv'

# Where cached covariates are stored
COVARIATES_FILE = 'covariates.csv'
COVARIATES_DRAW_FILE = 'covariates_draw_{draw_id}.csv'


# Columns from infectionator inputs
INFECTION_COL_DICT = {
    'COL_DATE': 'date',
    'COL_CASES': 'cases_draw',
    'COL_POP': 'pop',
    'COL_LOC_ID': 'loc_id',
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
