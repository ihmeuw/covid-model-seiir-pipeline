import pandas as pd

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_style('whitegrid')

from covid_model_seiir_pipeline.side_analysis.npi_location_splitting.data import (
    DataLoader,
)


def country_map(country_location_id: int,
                data_loader: DataLoader,
                map_data: pd.Series,
                hierarchy: pd.DataFrame,
                map_label: str,
                map_date: str,):
    # make output path before messing with map_date
    if data_loader.run_directory:
        output_path = data_loader.run_directory / 'diagnostics' / f'{country_name} {map_date} map.pdf'
    else:
        output_path = None

    # get populations
    populations = data_loader.load_populations()
    map_data = map_data.sort_index().dropna().groupby('location_id').cumsum()
    if map_date == 'last':
        map_data = map_data.drop(
            hierarchy.loc[
                (hierarchy['path_to_top_parent'].apply(lambda x: str(country_location_id) not in x.split(','))),
                'location_id'
            ],
            errors='ignore'
        )
        map_date = map_data.reset_index('date').groupby('location_id')['date'].last().min()
    else:
        map_date = pd.to_datetime(map_date)
    map_data = map_data.loc[:, map_date, :]
    map_data /= populations

    # identify admin level
    country_name = hierarchy.loc[hierarchy['location_id'] == country_location_id, 'location_name'].item()
    map_hierarchy = hierarchy.loc[
        (hierarchy['most_detailed'] == 1)
        & (hierarchy['path_to_top_parent'].apply(lambda x: str(country_location_id) in x.split(',')))
    ]
    admin_level = map_hierarchy.level.min() - 3
    map_location_ids = map_hierarchy['location_id'].to_list()

    # subset
    shapefile_data = gpd.read_file(data_loader.shapefile_root / f'lbd_standard_admin_{admin_level}_simplified.shx')
    shapefile_data = (shapefile_data
                      .loc[shapefile_data['loc_id'].isin(map_location_ids)]
                      .reset_index(drop=True)
                      .rename(columns= {'loc_id': 'location_id'})
                      .set_index('location_id'))
    map_data = shapefile_data.join(map_data, how='left')

    # clip at lower/upper bounds
    lb, ub = map_data['mapvar'].quantile([0.025, 0.975]).values
    map_data['mapvar'] = map_data['mapvar'].clip(lb, ub)
    n_bins = max(10, int(len(map_data) / 100))

    ## GENERATE FIGURE
    fig, ax = plt.subplots(1, 3,
                           gridspec_kw={'width_ratios': [12, 1, 2]}, figsize=(16, 8))
    fig.suptitle(f"{country_name},\n{map_label} {map_date.strftime('%Y-%m-%d')}")

    # map
    map_data.plot(ax=ax[0], column='mapvar', cmap='coolwarm',
                  # scheme='user_defined', classification_kwds={'bins':bins},
                  edgecolor='black', linewidth=0.2,
                  missing_kwds={'color':'lightgrey'},
                 )
    ax[0].set_axis_off()
    ax[0].get_figure()
    ax[0].axis('off')

    # colorbar
    mpl.colorbar.ColorbarBase(ax[1], cmap=mpl.cm.coolwarm,
                              norm=mpl.colors.Normalize(vmin=lb, vmax=ub))
    ax[1].yaxis.set_ticks_position('left')

    # histogram
    ax[2].hist(map_data['mapvar'], bins=n_bins,
               color='darkgrey', orientation='horizontal')
    ax[2].set_ylim(lb, ub)
    ax[2].axis('off')

    # print
    fig.tight_layout()
    if output_path:
        plt.savefig(output_path)
    else:
        fig.show()
