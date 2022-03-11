from typing import List

from loguru import logger
import numpy as np
import pandas as pd


def determine_mean_date_of_infection(location_dates: List,
                                     daily_infections: pd.Series,
                                     lag: int,) -> pd.DataFrame:
    dates_data = []
    for location_id, date in location_dates:
        data = daily_infections[location_id]
        data = data.reset_index()
        data = data.loc[data['date'] <= date - pd.Timedelta(days=lag)].reset_index(drop=True)
        if not data.empty:
            mean_infection_date_idx = int(np.round(np.average(data.index, weights=(data['daily_infections'] + 1))))
            try:
                mean_infection_date = data.loc[mean_infection_date_idx, 'date']
                dates_data.append(pd.DataFrame(
                    {'location_id': location_id,
                     'date': date,
                     'mean_infection_date': mean_infection_date + pd.Timedelta(days=lag)},
                    index=[0]
                ))
            except KeyError:
                logger.warning(
                    f'Encountered key error with key {mean_infection_date_idx} when '
                    f'determining mean date of infection for location {location_id}'
                )
    dates_data = pd.concat(dates_data).reset_index(drop=True)

    return dates_data
