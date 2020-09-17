import os

from setuptools import setup, find_packages


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, "src")

    about = {}
    with open(os.path.join(src_dir, "covid_model_seiir_pipeline", "__about__.py")) as f:
        exec(f.read(), about)

    with open(os.path.join(base_dir, "README.rst")) as f:
        long_description = f.read()

    install_requirements = [
        'click',
        'covid_shared>=1.0.32',
        'h5py',
        'loguru',
        'matplotlib',
        'numpy',
        'pandas',
        'pyyaml',
        'parse',
        'slime',
        'odeopt>=0.1.1'
    ]

    test_requirements = [
        'pytest',
        'pytest-mock',
    ]

    doc_requirements = []

    internal_requirements = [
        'jobmon',
        'db_queries',
    ]

    setup(
        name=about['__title__'],
        version=about['__version__'],

        description=about['__summary__'],
        long_description=long_description,
        license=about['__license__'],
        url=about["__uri__"],

        author=about["__author__"],
        author_email=about["__email__"],

        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        include_package_data=True,

        install_requires=install_requirements,
        tests_require=test_requirements,
        extras_require={
            'docs': doc_requirements,
            'test': test_requirements,
            'internal': internal_requirements,
            'dev': [doc_requirements, test_requirements, internal_requirements]
        },

        entry_points={'console_scripts': [
            'seiir=covid_model_seiir_pipeline.cli:seiir',
            'beta_regression=covid_model_seiir_pipeline.regression.task:main',
            'beta_forecast=covid_model_seiir_pipeline.forecasting.task.beta_forecast:main',
            'beta_residual_scaling=covid_model_seiir_pipeline.forecasting.task.beta_residual_scaling:main',
            'mean_level_mandate_reimposition=covid_model_seiir_pipeline.forecasting.task.mean_level_mandate_reimposition:main',
            'resample_map=covid_model_seiir_pipeline.forecasting.task.resample_map:main',
            'postprocess=covid_model_seiir_pipeline.forecasting.task.postprocessing:main',
            'oos_regression=covid_model_seiir_pipeline.predictive_validity.oos_regression:main',
            'oos_forecast=covid_model_seiir_pipeline.predictive_validity.oos_forecast:main',
        ]},
        zip_safe=False,
    )
