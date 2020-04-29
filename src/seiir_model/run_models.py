import os
import sys
import shutil

import pandas as pd

QSUB_STR = 'qsub -N {job_name} -P proj_covid -q d.q -b y -l m_mem_free=3G -l fthread=3 '\
           '{python} {model_file} '\
           '--data_folder {data_folder} --location {location} '\
           '--result_folder {result_folder}'


def submit_models(data_folder, result_folder):
    #data_folder.iterdir()
    locations = [loc.split('_')[2] for loc in os.listdir(data_folder) if loc.startswith('seir_data_')]

    for location in locations:
        location_clean = location.replace(' ', '___')
        if not os.path.exists(f'{result_folder}/{location_clean}'):
            os.mkdir(f'{result_folder}/{location_clean}')
        qsub_str = QSUB_STR.format(
            job_name=f'seir_{location_clean}',
            python=shutil.which('python'),
            model_file=f'{os.path.dirname(__file__)}/model.py',
            data_folder=data_folder,
            location=location_clean,
            result_folder=f'{result_folder}/{location_clean}'
        )

        job_str = os.popen(qsub_str).read()
        print(job_str)

if __name__ == '__main__':
    submit_models(sys.argv[1], sys.argv[2])
