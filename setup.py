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
        'pandas',
        'pyyaml',
        'matplotlib'
    ],
    entry_points={'console_scripts': [
        'run=seiir_model_pipeline.executor.run:main',
        'beta_regression=seiir_model_pipeline.executor.beta_regression:main',
        'beta_forecast=seiir_model_pipeline.executor.beta_forecast:main',
        'splice=seiir_model_pipeline.executor.splice:main',
        'create_regression_diagnostics=seiir_model_pipeline.executor.create_regression_diagnostics:main',
        'create_forecast_diagnostics=seiir_model_pipeline.executor.create_forecast_diagnostics:main',
        'create_scaling_diagnostics=seiir_model_pipeline.executor.create_scaling_diagnostics:main'
    ]},
    zip_safe=False,
)
