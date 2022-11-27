#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

from setuptools import find_packages, setup


def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join('omnisafe', '__init__.py'), 'r').read().split()
    return init[init.index('__version__') + 2][1:-1]


def get_install_requires() -> str:
    return [
        # TODO: 'torch>=1.10', different users may need different versions of torch.
        # TODO: It seems that the version of setuptools need to be specified.
        'tensorboard>=2.8.0',
        'mujoco==2.3.0',
        'numpy>=1.20.0',
        'psutil>=5.9.1',
        'scipy>=1.7.0',
        'joblib>=1.2.0',
        'pyyaml>=6.0',
        'xmltodict>=0.13.0',
        'setuptools~=59.5.0',
        'moviepy>=1.0.0',
        # test
        'pytest>=7.0.0',
        'pre-commit>=2.17.0',
        'isort>=5.10.0',
        'black>=22.1.0',
    ]


setup(
    name='omnisafe',
    version=get_version(),
    description='A comprehensive and reliable benchmark for safe reinforcement learning.',
    long_description=open('README.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/PKU-MARL/omnisafe',
    author='PKU-MARL',
    author_email='jiamg.ji@gmail.com',
    license='Apache-2.0 license',
    python_requires='>=3.8',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='Safe Reinforcement Learning Platform Pytorch',
    install_requires=get_install_requires(),
    py_modules=['omnisafe'],
)
