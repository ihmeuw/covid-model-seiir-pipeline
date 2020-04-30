from setuptools import setup
from setuptools import find_packages

setup(
    name='seiir_model_pipeline',
    version='0.0.0',
    description='SEIIR model',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'argparse',
        'mrtool',
        'numpy',
        'odeopt',
        'pandas'
    ],
    entry_points={'console_scripts': [
        'run=seiir_model_pipeline.executor.run:main',
        'run_draw=seiir_model_pipeline.executor.run_one_draw:main'
    ]},
    zip_safe=False,
)
