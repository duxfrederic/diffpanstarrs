#!/usr/bin/env python

from setuptools import setup, find_packages
with open("README.md", 'r') as f:
    long_description = f.read()
    
setup(
    name='diffpanstarrs',
    version='0.11',
    python_requires='>3.7.0',
    description='Downloads Pan-STARRS data around given coordinates and performs difference imaging to extract light curves',
    author='Frederic Dux, Cameron Lemon',
    author_email='frederic.dux@epfl.ch',
    long_description=long_description,
    license='GPL',
    packages=find_packages(),
    package_data={
        "diffpanstarrs": ["sextractor_defaults/*.param", "sextractor_defaults/*.conv", 'cnn_weights/model.hdf5'],
        #"tests": ["test_data/*"]
    },
    install_requires=[
        'urllib3',
        'wget',
        'numpy',
        'scipy',
        'astropy',
        'matplotlib',
        'scikit-image',
        'robustats'
        ],
    entry_points = {
        'console_scripts': ['plot_this_directory=diffpanstarrs.command_line:plot_this_directory']
        }
)
