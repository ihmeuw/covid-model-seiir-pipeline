# -*- coding: utf-8 -*-
"""
    Runner script for the ODE optimization to obtain the estimation of the beta.
"""
import argparse
import json
import numpy as np
import pandas as pd
from odeopt.ode import RK4
from .ode_process import SingleGroupODEProcess


class SingleGroupODEPipeline:
    """BetaSEIIR Single Group ODE Pipline"""
    def __init__(self, df,
                 col_date,
                 col_cases,
                 col_pop,
                 col_loc_id,
                 alpha,
                 sigma,
                 gamma1,
                 gamma2):
        """Constructor for the Single Group ODE Pipeline

        Args:
            col_date (str): Column with date.
            col_cases (list{str}):
                Columns with cases (potential many draws).
            col_pop (str): Column with total population.
            col_loc_id (str): Column with location id.
            alpha (arraylike): bounds for uniformly sampling alpha.
            sigma (arraylike): bounds for uniformly sampling sigma.
            gamma1 (arraylike): bounds for uniformly sampling gamma1.
            gamma2 (arraylike): bounds for uniformly sampling gamma2.
        """
        assert col_date in df
        assert col_pop in df
        assert col_loc_id in df
        assert all([c in df for c in col_cases])
        self.col_date = col_date
        self.col_cases = col_cases
        self.col_pop = col_pop
        self.col_loc_id = col_loc_id

        # process data
        df[self.col_date] = pd.to_datetime(df[self.col_date])
        df.sort_values(self.col_date, inplace=True)

        self.N = df[self.col_pop].values[0]
        self.loc_id = df[self.col_loc_id].values[0]
        self.date = pd.to_datetime(df[self.col_date])
        self.days = (df[self.col_date] -
                     df[self.col_date].min()).dt.days.values
        self.cases = np.ascontiguousarray(df[self.col_cases].values.T)

        # process parameter
        self.alpha = np.array(alpha)
        self.sigma = np.array(sigma)
        self.gamma1 = np.array(gamma1)
        self.gamma2 = np.array(gamma2)
        assert self.alpha.size == 2 and \
            0.0 <= self.alpha[0] <= self.alpha[1]
        assert self.sigma.size == 2 and \
            0.0 <= self.sigma[0] <= self.sigma[1]
        assert self.gamma1.size == 2 and \
            0.0 <= self.gamma1[0] <= self.gamma1[1]
        assert self.gamma2.size == 2 and \
            0.0 <= self.gamma2[0] <= self.gamma2[1]

        self.num_sub_models = len(self.col_cases)
        self.params = np.hstack([
            np.random.uniform(*self.alpha, self.num_sub_models)[:, None],
            np.random.uniform(*self.sigma, self.num_sub_models)[:, None],
            np.random.uniform(*self.gamma1, self.num_sub_models)[:, None],
            np.random.uniform(*self.gamma2, self.num_sub_models)[:, None],
        ])

        self.create_process()

    def create_process(self):
        self.process = SingleGroupODEProcess(
            self.days,
            self.cases[0],
            *self.params[0], self.N,
            spline_options={
                'spline_degree': 3,
                'spline_knots': np.linspace(0.0, 1.0, 7)
            },
            solver_class=RK4,
            solver_dt=1e-1)

    def run(self):
        print(f'loc_id: {self.loc_id}')
        print('\t', f'there are {self.num_sub_models} sub-models')
        self.process.fit_spline()
        self.beta = []
        for i, param in enumerate(self.params):
            print('\t', f'progress: {(i+1)/self.num_sub_models:.2f}', end='\r')
            self.process.update_data(self.cases[i])
            self.process.update_ode_params(alpha=param[0],
                                           sigma=param[1],
                                           gamma1=param[2],
                                           gamma2=param[3])
            self.process.process(fit_spline=False)
            self.beta.append(self.process.predict(self.days)[0][0])

        print('\t', f'progress: done')
        df_result = pd.DataFrame({
            'loc_id': self.loc_id,
            'date': self.date,
            'days': self.days,
        })
        for i in range(self.num_sub_models):
            df_result[f'draw_{i}'] = self.beta[i]

        return df_result


def main():
    '''
    args = argparse.Namespace(
        input_dir='/ihme/covid-19/seir-inputs/2020_04_25.04',
        location_id=523,
        location_map_path='/ihme/homes/rmbarber/covid-19/seir_test_2020_04_29/_location_map.json',
        output_path='/ihme/homes/rmbarber/covid-19/seir_test_2020_04_28/523.csv'
    )
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', help='Name of location-specific input file.', type=str
    )
    parser.add_argument(
        '--location_id', help='`location_id` for job.', type=str
    )
    parser.add_argument(
        '--location_map_path', help='File that contains location name mapping.', type=str
    )
    parser.add_argument(
        '--output_path', help='Name of location-specific output file.', type=str
    )
    args = parser.parse_args()

    # identify location file naming
    with open(args.location_map_path, 'r') as fread:
        location_map = json.load(fread)
    location_dir = location_map[str(args.location_id)]['location_dir']
    file_name = location_map[str(args.location_id)]['file_name']

    # read data frame
    df = pd.read_csv(f'{args.input_dir}/{location_dir}/{file_name}')

    # run pipeline
    pipeline = SingleGroupODEPipeline(
        df,
        'date',
        [col for col in df if col.startswith('case_draw')],
        'pop',
        'loc_id',
        [0.9, 1.0],
        [1/6, 1/4],
        [0.5, 0.5],
        [1/3, 1/2]
    )
    df_result = pipeline.run()
    df_result = df_result.rename(index=str, columns={'loc_id':'location_id'})

    # store results
    df_result.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    main()
