#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(
    name='boardgame2',
    version='2.0.0',
    description='2-player zero-sum board game extension for OpenAI Gym.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Zhiqing Xiao',
    author_email='xzq.xiaozhiqing@gmail.com',
    python_requires='>=3.9.0',
    url='http://github.com/zhiqingxiao/boardgame2/',
    packages=find_packages(),
    install_requires=['six', 'numpy', 'gym>=0.26'],
    extras_require={},
    test_require=['pytest'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Operating System :: OS Independent',
        'Topic :: Games/Entertainment',
        'Topic :: Games/Entertainment :: Board Games',
    ],
)
