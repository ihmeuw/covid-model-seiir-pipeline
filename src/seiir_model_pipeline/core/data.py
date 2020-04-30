import pandas as pd


def load_all_location_data(directories, location_ids):
    dfs = dict()
    for loc in location_ids:
        file = directories.get_infection_file(location_id=loc, input_dir=directories.infection_dir)
        dfs[loc] = pd.read_csv(file)
    return dfs

def load_peaked_dates(dir, col_loc_id, col_date):
    df = pd.read_csv(dir)
    return dict(zip(df[col_loc_id], df[col_date]))
