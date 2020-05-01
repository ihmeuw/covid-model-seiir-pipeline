import pandas as pd
from seiir_model.ode_model import ODEProcessInput
from seiir_model_pipeline.core.versioner import INFECTION_COL_DICT

from slime.model import CovModel, CovModelSet
from seiir_model_pipeline.core.versioner import PEAK_DATE_FILE
from slime.core.data import MRData

SEIIR_COMPARTMENTS = ['S', 'E', 'I1', '12', 'R']

COL_T = 'days'
COL_BETA = 'beta'
COL_GROUP = 'loc_id'

LOCATION_SET_ID = 111


def get_locations(location_metadata_file):
    df = pd.read_csv(location_metadata_file)
    return df.location_id.unique()


def get_peaked_dates_from_file():
    df = pd.read_csv(PEAK_DATE_FILE)
    return dict(zip(df.location_id, df.peak_date))


def process_ode_process_input(settings, location_data, peak_data):
    """Convert to ODEProcessInput.
    """
    import pdb; pdb.set_trace()
    return ODEProcessInput(
        df_dict=location_data,
        col_date=INFECTION_COL_DICT['COL_DATE'],
        col_cases=INFECTION_COL_DICT['COL_CASES'],
        col_pop=INFECTION_COL_DICT['COL_POP'],
        col_loc_id=INFECTION_COL_DICT['COL_LOC_ID'],
        alpha=tuple(settings['alpha']),
        sigma=tuple(settings['sigma']),
        gamma1=tuple(settings['gamma1']),
        gamma2=tuple(settings['gamma2']),
        peak_date_dict=peak_data,
        day_shift=settings['day_shift'],
        solver_dt=settings['solver_dt'],
        spline_options={
            'spline_knots': settings['knots'],
            'spline_degree': settings['degree']
        }
    )


def convert_to_covmodel(cov_dict):
    cov_models = []
    for name, dct in cov_dict.items():
        cov_models.append(CovModel(
            name,
            use_re=dct['use_re'],
            bounds=dct['bounds'],
            gprior=dct['gprior'],
            re_var=dct['re_var'],
        ))
    covmodel_set = CovModelSet(cov_models)
    return covmodel_set


def convert_inputs_for_beta_model(data_cov, df_beta, covmodel_set):
    df_cov, col_t_cov, col_group_cov = data_cov
    df = df_beta.merge(
        df_cov,
        left_on=[COL_T, COL_GROUP],
        right_on=[col_t_cov, col_group_cov],
    ).copy()
    df.sort_values(inplace=True, by=[COL_GROUP, COL_T])
    cov_names = [covmodel.col_cov for covmodel in covmodel_set.cov_models]
    mrdata = MRData(df, col_group=COL_GROUP, col_obs=COL_BETA, col_covs=cov_names)

    return mrdata
