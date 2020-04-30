import os
import sys
import shutil
import json
import pandas as pd

QSUB_STR = 'qsub -N {job_name} -P proj_covid -q d.q -b y -l m_mem_free=3G -l fthread=3 '\
           '{python} {model_file} '\
           '--input_dir {input_dir} --location_id {location_id} '\
           '--location_map_path {location_map_path} --output_path {output_path}'


def submit_models(input_dir, output_dir):
    # get location sub-dirs
    loc_dirs = [loc_dir for loc_dir in os.listdir(input_dir) if os.path.isdir(f'{input_dir}/{loc_dir}')]
    for loc_dir in loc_dirs:
        if not os.path.exists(f'{input_dir}/{loc_dir}/prepped_deaths_and_cases_all_age.csv'):
            raise ValueError('Incompatible path.')

    # store map from location_id to draw file
    location_map = dict()
    location_ids = [int(loc_dir.split('_')[1]) for loc_dir in loc_dirs]
    for location_id, loc_dir in zip(location_ids, loc_dirs):
        location_map.update({
            location_id: {'location_dir': loc_dir,
                          'file_name':'prepped_deaths_and_cases_all_age.csv'}
        })
    with open(f'{output_dir}/_location_map.json', 'w') as fwrite:
        json.dump(location_map, fwrite)

    # submit jobs
    for location_id in location_ids:
        qsub_str = QSUB_STR.format(
            job_name=f'seiir_{location_id}',
            python=shutil.which('python'),
            model_file=f'{os.path.dirname(__file__)}/model.py',
            input_dir=input_dir,
            location_id=location_id,
            location_map_path=f'{output_dir}/_location_map.json',
            output_path=f'{output_dir}/{location_id}.csv'
        )

        job_str = os.popen(qsub_str).read()
        print(job_str)

if __name__ == '__main__':
    submit_models(sys.argv[1], sys.argv[2])
