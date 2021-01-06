from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PyPDF2 import PdfFileMerger


def merge_pdfs(plot_cache: Path, output_path: Path, locs_to_plot: List[int], hierarchy: pd.DataFrame):
    parent_map = hierarchy.set_index('location_id').parent_id
    name_map = hierarchy.set_index('location_id').location_ascii_name

    sorted_locations = get_locations_dfs(hierarchy)

    merger = PdfFileMerger()
    try:
        current_page = 0
        for location_id in sorted_locations:
            result_page_path = plot_cache / f'{location_id}_results.pdf'
            covariate_page_path = plot_cache / f'{location_id}_results.pdf'

            if not result_page_path.exists():
                continue
            
            merger.merge(current_page, str(result_page_path))
            
            if parent_map[location_id] != location_id:
                parent = name_map.loc[parent_map.loc[location_id]]
            else:
                parent = None
            merger.addBookmark(name_map.loc[location_id], current_page, parent)
            merger.merge(current_page + 1, str(covariate_page_path))
            current_page += 2

        if output_path.exists():
            output_path.unlink()
        merger.write(str(output_path))
    finally:
        merger.close()


def get_locations_dfs(hierarchy: pd.DataFrame) -> List[int]:    
    def _get_locations(location: pd.Series):
        locs = [location.location_id]
        
        children = hierarchy[(hierarchy.parent_id == location.location_id)
                             & (hierarchy.location_id != location.location_id)]
        for child in children.sort_values('location_ascii_name').itertuples():
            locs.extend(_get_locations(child))
        return locs
    
    top_locs = hierarchy[hierarchy.location_id == hierarchy.parent_id]
    locations = []
    for top_loc in top_locs.sort_values('location_ascii_name').itertuples():
        locations.extend(_get_locations(top_loc))

    return locations

