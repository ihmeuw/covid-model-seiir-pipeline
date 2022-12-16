import pandas as pd

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
sns.set_style('whitegrid')

from covid_model_seiir_pipeline.side_analysis.npi_location_splitting.data import (
    DataLoader,
)
from covid_model_seiir_pipeline.side_analysis.npi_location_splitting.model import (
    get_prod_location_label,
)


def country_map(
    country_location_id: int,
    data_loader: DataLoader,
    map_data: pd.Series,
    npi_hierarchy: pd.DataFrame,
    map_label: str,
    map_date: str,
    convert_to_rate: bool = True,
):
    # get country name first for file name
    country_name = npi_hierarchy.loc[npi_hierarchy['location_id'] == country_location_id, 'location_name'].item()

    # make output path before messing with map_date
    if data_loader.run_directory:
        output_path = data_loader.run_directory / 'diagnostics' / f'{country_name} {map_date} map.pdf'
    else:
        output_path = None

    # subset to specified date
    if map_date == 'last':
        map_data = map_data.drop(
            npi_hierarchy.loc[
                (npi_hierarchy['path_to_top_parent'].apply(lambda x: str(country_location_id) not in x.split(','))),
                'location_id'
            ],
            errors='ignore'
        )
        map_date = map_data.reset_index('date').groupby('location_id')['date'].last().min()
    else:
        map_date = pd.to_datetime(map_date)
    map_data = map_data.loc[:, map_date, :]

    # if specified, convert data to rate
    if convert_to_rate:
        populations = data_loader.load_populations()
        map_data /= populations

    # identify admin level
    map_hierarchy = npi_hierarchy.loc[
        (npi_hierarchy['most_detailed'] == 1)
        & (npi_hierarchy['path_to_top_parent'].apply(lambda x: str(country_location_id) in x.split(',')))
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
                           gridspec_kw={'width_ratios': [12, 1, 2]}, figsize=(15, 8))
    fig.suptitle(f"{country_name},\n{map_label} {map_date.strftime('%Y-%m-%d')}")

    # map
    map_data.plot(ax=ax[0], column='mapvar', cmap='coolwarm',
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
        plt.close()
    else:
        fig.show()


def aggregate_infections_time_series(
    data_loader: DataLoader,
    npi_hierarchy: pd.DataFrame,
    prod_hierarchy: pd.DataFrame,
    npi_infections: pd.DataFrame,
    prod_infections: pd.Series,
):
    # make output path before messing with map_date
    if data_loader.run_directory:
        output_path = data_loader.run_directory / 'diagnostics' / 'aggregate infections time series.pdf'
    else:
        output_path = None

    agg_npi_infections = (get_prod_location_label(npi_hierarchy, prod_hierarchy)
                          .join(npi_infections, how='right')
                          .reset_index())
    agg_location_ids = (agg_npi_infections
                        .loc[agg_npi_infections['location_id'] != agg_npi_infections['prod_location_id'],
                             'prod_location_id']
                        .drop_duplicates()
                        .to_list())
    agg_npi_infections = (agg_npi_infections
                          .drop('location_id', axis=1)
                          .groupby(['prod_location_id', 'date']).sum())

    locations = npi_hierarchy.set_index('location_id').loc[agg_location_ids].sort_values('sort_order').loc[:, 'location_name']

    if output_path:
        with PdfPages(output_path) as pdf:
            for location_id, location_name in locations.items():
                _aggregate_infections_time_series(
                    agg_npi_infections.loc[location_id],
                    prod_infections.loc[location_id],
                    location_name,
                    pdf,
                )
    else:
        for location_id, location_name in locations.items():
            _aggregate_infections_time_series(
                agg_npi_infections.loc[location_id],
                prod_infections.loc[location_id],
                location_name,
            )


def _aggregate_infections_time_series(
    agg_npi_infections: pd.DataFrame,
    prod_infections: pd.Series,
    location_name: str,
    pdf: PdfPages = None,
):
    fig, ax = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    (agg_npi_infections / 1e3).plot(ax=ax[0])
    (prod_infections.reindex(agg_npi_infections.index) / 1e3).plot(ax=ax[0],
                                                                   linestyle='--',
                                                                   color='darkgrey')

    (agg_npi_infections.cumsum() / 1e6).plot(ax=ax[1])
    (prod_infections.reindex(agg_npi_infections.index).cumsum() / 1e6).plot(ax=ax[1],
                                                                             linestyle='--',
                                                                             color='darkgrey')

    ax[0].set(xlabel=None, ylabel='Daily infections (thousands)')
    ax[1].set(xlabel=None, ylabel='Cumulative infections (millions)')
    fig.suptitle(location_name)

    fig.tight_layout()
    if pdf:
        pdf.savefig(fig)
        plt.close()
    else:
        fig.show()
