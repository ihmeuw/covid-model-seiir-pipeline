from pathlib import Path

from setuptools import setup, find_packages


if __name__ == "__main__":

    base_dir = Path(__file__).parent
    src_dir = base_dir / 'src'

    about = {}
    with (src_dir / "covid_model_seiir_pipeline" / "__about__.py").open() as f:
        exec(f.read(), about)

    with (base_dir / "README.rst").open() as f:
        long_description = f.read()

    install_requirements = [
        'matplotlib',
        'numpy',
        'pandas',
        'pyyaml',
        'matplotlib',
        'slime',
        'odeopt>=0.1.1',
    ]

    test_requirements = [
        'pytest',
        'pytest-mock',
    ]

    doc_requirements = []

    internal_requirements = [
        'jobmon',
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
        entry_points = {'console_scripts': [
            'run=covid_model_seiir_pipeline.executor.run:main',
            'beta_ode_fit=covid_model_seiir_pipeline.executor.beta_ode_fit:main',
            'beta_regression=covid_model_seiir_pipeline.executor.beta_regression:main',
            'beta_forecast=covid_model_seiir_pipeline.executor.beta_forecast:main',
            'splice=covid_model_seiir_pipeline.executor.splice:main',
            'create_regression_diagnostics=covid_model_seiir_pipeline.executor.create_regression_diagnostics:main',
            'create_forecast_diagnostics=covid_model_seiir_pipeline.executor.create_forecast_diagnostics:main',
            'create_scaling_diagnostics=covid_model_seiir_pipeline.executor.create_scaling_diagnostics:main',
            'run_validation=covid_model_seiir_pipeline.executor.run_validation:main',
            'split_infectionator=covid_model_seiir_pipeline.executor.split_infectionator:main',
            'evaluate_location=covid_model_seiir_pipeline.executor.evaluate_location:main',
            'analyze_validation_output=covid_model_seiir_pipeline.executor.analyze_validation_output:main'
        ]},
        zip_safe=False,
    )
