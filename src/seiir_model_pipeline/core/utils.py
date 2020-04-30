import pandas as pd
from seiir_model.ode_model import ODEProcessInput
from seiir_model_pipeline.core.file_master import INFECTION_COL_DICT


def get_peaked_dates_from_file(peak_date_file):
    df = pd.read_csv(peak_date_file)
    return dict(zip(df.location_id, df.peak_date))


def process_ode_process_input(settings, location_data, peak_data):
    """Convert to ODEProcessInput.
    """
    return ODEProcessInput(
        df_dict=location_data,
        col_date=INFECTION_COL_DICT['COL_DATE'],
        col_cases=INFECTION_COL_DICT['COL_CASES'],
        col_pop=INFECTION_COL_DICT['COL_POP'],
        col_loc_id=INFECTION_COL_DICT['loc_id'],
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
