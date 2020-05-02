import pandas as pd
from seiir_model_pipeline.core.versioner import INFECTION_COL_DICT


class Splicer:
    def __init__(self):
        self.col_loc_id = INFECTION_COL_DICT['COL_LOC_ID']
        self.col_date = INFECTION_COL_DICT['COL_DATE']

    def splice_draw(self, infection_data, component_data):
        # DO THE SPLICING!
        return pd.DataFrame()
