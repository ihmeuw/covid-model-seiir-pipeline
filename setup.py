from setuptools import setup
from setuptools import find_packages

setup(
    name='seiir_model',
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
    zip_safe=False,
)
