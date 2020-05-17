import os

from setuptools import setup, find_packages


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, "src")

    about = {}
    with open(os.path.join(src_dir, "seiir_model_pipeline", "__about__.py")) as f:
        exec(f.read(), about)

    with open(os.path.join(base_dir, "README.md")) as f:
        long_description = f.read()

    install_requirements = [
        'click',
        'covid_shared>=1.0.15',
        'loguru',
        'matplotlib',
        'mrtool',
        'numpy',
        'odeopt',
        'pandas',
        'pyyaml',
    ]

    test_requirements = [
        'pytest',
        'pytest-mock',
    ]

    doc_requirements = [
        'sphinx',
    ]

    setup(
        name=about['__title__'],
        version=about['__version__'],

        description=about['__summary__'],
        long_description=long_description,
        url=about["__uri__"],

        author=about["__author__"],
        author_email=about["__email__"],

        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        include_package_data=True,

        install_requires=install_requirements,
        extras_require={
            'test': test_requirements,
            'dev': test_requirements + doc_requirements,
            'docs': doc_requirements
        },

        entry_points={'console_scripts': [
            'seiir=seiir_model_pipeline.cli:seiir',
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
