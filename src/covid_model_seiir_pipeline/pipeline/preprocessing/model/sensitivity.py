from typing import Dict, List

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    math,
    utilities,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.data import (
    PreprocessingDataInterface,
)

logger = cli_tools.task_performance_logger


def preprocess_sensitivity(data_interface: PreprocessingDataInterface) -> None:
    logger.info('Loading sensitivity input data', context='read')
    raw_sensitivity_data = data_interface.load_sensitivity_data()

    logger.info('Cleaning sensitivity input data', context='transform')
    sensitivity_data = format_assay_sensitivity(raw_sensitivity_data)
    sensitivity_samples = sample_sensitivity(
        sensitivity_data,
        n_samples=data_interface.get_n_draws(),
    )
    logger.info('Writing sensitivity data.', context='write')
    data_interface.save_sensitivity(sensitivity_data)
    for draw, sample in enumerate(sensitivity_samples):
        data_interface.save_sensitivity(sample, draw_id=draw)


def format_assay_sensitivity(sensitivity: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    peluso = sensitivity['peluso']
    peluso['t'] = peluso['Time'].apply(lambda x: int(x.split(' ')[0]) * 30)
    peluso['sensitivity_std'] = (peluso['97.5%'] - peluso['2.5%']) / 3.92
    peluso = peluso.rename(columns={'mean': 'sensitivity_mean',
                                    'AntigenAndAssay': 'assay',
                                    'Hospitalization_status': 'hospitalization_status' ,})
    peluso = peluso.loc[:, ['assay', 'hospitalization_status', 't', 'sensitivity_mean', 'sensitivity_std']]
    # only need to keep commercial assays
    peluso = peluso.loc[~peluso['assay'].isin(['Neut-Monogram', 'RBD-LIPS', 'RBD-Split Luc',
                                               'RBD-Lum', 'S-Lum', 'N(full)-Lum', 'N-LIPS',
                                               'N(frag)-Lum', 'N-Split Luc'])]
    peluso['source'] = 'Peluso'

    ## PEREZ-SAEZ - start at 21 days out
    perez_saez = sensitivity['perez_saez']
    perez_saez['metric'] = perez_saez['metric'].str.strip()
    perez_saez = pd.pivot_table(perez_saez, index=['t', 'assay'], columns='metric', values='sensitivity').reset_index()
    perez_saez = perez_saez.rename(columns={'mean': 'sensitivity_mean'})
    perez_saez['sensitivity_std'] = (perez_saez['upper'] - perez_saez['lower']) / 3.92
    perez_saez = perez_saez.loc[perez_saez['t'] >= 21]
    perez_saez['t'] -= 21
    perez_saez = pd.concat([
        pd.concat([perez_saez, pd.DataFrame({'hospitalization_status': 'Non-hospitalized'}, index=perez_saez.index)], axis=1),
        pd.concat([perez_saez, pd.DataFrame({'hospitalization_status': 'Hospitalized'}, index=perez_saez.index)], axis=1)
    ])
    perez_saez['source'] = 'Perez-Saez'

    ## BOND - drop 121-150 point, is only 11 people and can't possibly be at 100%; start 21 days out; only keep Abbott
    bond = sensitivity['bond']
    bond = bond.loc[bond['days since symptom onset'] != '121–150']
    bond['t_start'] = bond['days since symptom onset'].str.split('–').apply(lambda x: int(x[0]))
    bond['t_end'] = bond['days since symptom onset'].str.split('–').apply(lambda x: int(x[1]))
    bond['t'] = bond[['t_start', 't_end']].mean(axis=1)
    for assay in ['N-Abbott', 'S-DiaSorin', 'N-Roche']:
        bond[f'{assay} mean'] = bond[assay].str.split('%').apply(lambda x: float(x[0])) / 100
        bond[f'{assay} lower'] = bond[assay].str.split('% \[').apply(lambda x: float(x[1].split(', ')[0])) / 100
        bond[f'{assay} upper'] = bond[assay].str.split('% \[').apply \
            (lambda x: float(x[1].split(', ')[1].replace(']', ''))) / 100
        bond[f'{assay} std'] = (bond[f'{assay} upper'] - bond[f'{assay} lower']) / 3.92
    bond_mean = pd.melt(bond, id_vars='t', value_vars=['N-Abbott mean', 'S-DiaSorin mean', 'N-Roche mean'],
                        var_name='assay', value_name='sensitivity_mean')
    bond_mean['assay'] = bond_mean['assay'].str.replace(' mean', '')
    bond_std = pd.melt(bond, id_vars='t', value_vars=['N-Abbott std', 'S-DiaSorin std', 'N-Roche std'],
                       var_name='assay', value_name='sensitivity_std')
    bond_std['assay'] = bond_std['assay'].str.replace(' std', '')
    bond = bond_mean.merge(bond_std)
    bond = bond.loc[bond['t'] >= 21]
    bond['t'] -= 21
    bond = pd.concat([
        pd.concat([bond, pd.DataFrame({'hospitalization_status': 'Non-hospitalized'}, index=bond.index)], axis=1),
        pd.concat([bond, pd.DataFrame({'hospitalization_status': 'Hospitalized'}, index=bond.index)], axis=1)
    ])
    bond = bond.loc[bond['assay'] == 'N-Abbott']
    bond['source'] = 'Bond'

    ## MUECKSCH - top end of terminal group is 110 days; only keep Abbott
    muecksch = sensitivity['muecksch']
    muecksch.loc[muecksch['Time, d'] == '>81', 'Time, d'] = '81-110'
    muecksch['t_start'] = muecksch['Time, d'].str.split('-').apply(lambda x: int(x[0]))
    muecksch['t_end'] = muecksch['Time, d'].str.split('-').apply(lambda x: int(x[1]))
    muecksch['t'] = muecksch[['t_start', 't_end']].mean(axis=1)
    for assay in ['N-Abbott', 'S-DiaSorin', 'RBD-Siemens']:
        muecksch[f'{assay} mean'] = muecksch[assay].str.split(' ').apply(lambda x: float(x[0])) / 100
        muecksch[f'{assay} lower'] = muecksch[assay].str.split(' \[').apply(lambda x: float(x[1].split('-')[0])) / 100
        muecksch[f'{assay} upper'] = (muecksch[assay].str.split(' \[')
                                      .apply(lambda x: float(x[1].split('-')[1].replace(']', ''))) / 100)
        muecksch[f'{assay} std'] = (muecksch[f'{assay} upper'] - muecksch[f'{assay} lower']) / 3.92
    muecksch_mean = pd.melt(muecksch, id_vars='t', value_vars=['N-Abbott mean', 'S-DiaSorin mean', 'RBD-Siemens mean'],
                            var_name='assay', value_name='sensitivity_mean')
    muecksch_mean['assay'] = muecksch_mean['assay'].str.replace(' mean', '')
    muecksch_std = pd.melt(muecksch, id_vars='t', value_vars=['N-Abbott std', 'S-DiaSorin std', 'RBD-Siemens std'],
                           var_name='assay', value_name='sensitivity_std')
    muecksch_std['assay'] = muecksch_std['assay'].str.replace(' std', '')
    muecksch = muecksch_mean.merge(muecksch_std)
    muecksch['t'] -= 24
    muecksch = pd.concat([
        pd.concat([muecksch, pd.DataFrame({'hospitalization_status': 'Non-hospitalized'}, index=muecksch.index)], axis=1),
        pd.concat([muecksch, pd.DataFrame({'hospitalization_status': 'Hospitalized'}, index=muecksch.index)], axis=1)
    ])
    muecksch = muecksch.loc[muecksch['assay'] == 'N-Abbott']
    muecksch['source'] = 'Muecksch'

    ## LUMLEY
    lumley = sensitivity['lumley']
    lumley['metric'] = lumley['metric'].str.strip()
    lumley = pd.pivot_table(lumley, index=['t', 'assay', 'num_60', 'denom_60', 'avg_60'],
                            columns='metric', values='sensitivity').reset_index()
    lumley = lumley.rename(columns={'mean': 'sensitivity_mean'})
    lumley['sensitivity_std'] = (lumley['upper'] - lumley['lower']) / 3.92
    lumley['sensitivity_mean'] *= (lumley['num_60'] / lumley['denom_60']) / lumley['avg_60']
    lumley = pd.concat([
        pd.concat([lumley, pd.DataFrame({'hospitalization_status': 'Non-hospitalized'}, index=lumley.index)], axis=1),
        pd.concat([lumley, pd.DataFrame({'hospitalization_status': 'Hospitalized'}, index=lumley.index)], axis=1)
    ])
    lumley['source'] = 'Lumley'

    # combine them all
    keep_cols = ['source', 'assay', 'hospitalization_status', 't', 'sensitivity_mean', 'sensitivity_std']
    sensitivity = pd.concat([peluso.loc[:, keep_cols],
                             perez_saez.loc[:, keep_cols],
                             bond.loc[:, keep_cols],
                             muecksch.loc[:, keep_cols],
                             lumley.loc[:, keep_cols]]).reset_index(drop=True)
    return sensitivity


def sample_sensitivity(sensitivity_data: pd.DataFrame,
                       n_samples: int,
                       floor: float = 1e-4,
                       logit_se_cap: float = 1.) -> List[pd.DataFrame]:
    logit_mean = math.logit(sensitivity_data['sensitivity_mean'].clip(floor, 1 - floor))
    logit_sd = (sensitivity_data['sensitivity_std'] / (logit_mean * (1 - logit_mean))).clip(0, logit_se_cap)

    random_state = utilities.get_random_state('sample_seroreversion_error')
    logit_samples = random_state.normal(loc=logit_mean.to_frame().values,
                                        scale=logit_sd.to_frame().values,
                                        size=(len(sensitivity_data), n_samples), )
    samples = math.expit(logit_samples)

    ## CANNOT DO THIS, MOVES SOME ABOVE 1
    # # re-center around original mean
    # samples *= sensitivity_data[['sensitivity_mean']].values / samples.mean(axis=1, keepdims=True)

    # sort
    samples = np.sort(samples, axis=1)

    sample_list = []
    for sample in samples.T:
        _sample = sensitivity_data.drop(['sensitivity_mean', 'sensitivity_std', ], axis=1).copy()
        _sample['sensitivity'] = sample
        sample_list.append(_sample.reset_index(drop=True))

    return sample_list
