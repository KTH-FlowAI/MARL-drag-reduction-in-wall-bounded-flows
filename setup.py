#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='MARL_simson',
    version='0.0.1',
    author="Luca Guastoni",
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    long_description=open('README.md').read(),
)
