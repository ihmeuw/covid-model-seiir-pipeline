from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PyPDF2 import PdfFileMerger


def merge_pdfs(plot_cache: Path, output_path: Path, locs_to_plot: List[int], hierarchy: pd.DataFrame):
    parent_map = hierarchy.set_index('location_id').parent_id
    name_map = hierarchy.set_index('location_id').location_ascii_name

    page_mapping = get_pages_dfs(hierarchy)

    merger = PdfFileMerger()
    try:
        for location_id in locs_to_plot:
            page = page_mapping[location_id]
            merger.merge(page['page'], str(plot_cache / f'{location_id}_results.pdf'))
            if page['parent_id']:
                parent = name_map.loc[parent_map.loc[location_id]]
            else:
                parent = None
            merger.addBookmark(name_map.loc[location_id], page['page']+1, parent)
            merger.merge(page['page'] + 1, str(plot_cache / f'{location_id}_covariates.pdf'))

        if output_path.exists():
            output_path.unlink()
        merger.write(str(output_path))
    finally:
        merger.close()


def get_pages_dfs(hierarchy: pd.DataFrame) -> Dict[int, Dict[str, Optional[int]]]:
    top_locs = hierarchy[hierarchy.location_id == hierarchy.parent_id]
    mapping = {}

    def _get_page(location: pd.Series, page: int):
        if location.location_id == location.parent_id:
            parent = None
        else:
            parent = location.parent_id
        mapping[location.location_id] = {
            'page': page,
            'parent_id': parent
        }
        children = hierarchy[(hierarchy.parent_id == location.location_id)
                             & (hierarchy.location_id != location.location_id)]
        for child in children.sort_values('location_id').itertuples():
            page = _get_page(child, page + 2)
        return page

    page = -2
    for top_loc in top_locs.sort_values('location_id').itertuples():
        page = _get_page(top_loc, page + 2)

    return mapping

