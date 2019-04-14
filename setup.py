#!/usr/bin/env python
"""
This is an example for the possible setup of a library
located by default within the src/ folder.
All packages will be installed to python.site-packages
simply run:

    >>> python setup.py install

For a local installation or if you like to develop further

    >>> python setup.py develop --user


The test_suite located within the test/ folder
will be executed automatically.
"""
import codecs
from setuptools import setup, find_packages
import sys
import os

source_path = 'src'
path = os.path.abspath(os.path.dirname(sys.argv[0]))
os.chdir(path)
packages = find_packages(source_path)


def get_version():
    """
    get package version

    :return:
    """
    with open('VERSION.md') as version_file:
        __version__ = version_file.read().strip()
    return __version__


def read_files(*filenames):
    """
    Output the contents of one or more files to a single concatenated string.
    """
    output = []
    for filename in filenames:
        f = codecs.open(filename, encoding='utf-8')
        try:
            output.append(f.read())
        finally:
            f.close()
    return '\n\n'.join(output)


setup(
    name='marksim',
    version=get_version(),
    description='Detects anomalies in the unbalanced panel data using markov chain simulations',
    long_description=read_files('README.md'),
    author='Vladimir Korzinov',
    author_email='korzinovvv@gmail.com',
    url='',
    packages=packages,
    install_requires=[],
    extras_require={
        'about-page':  ["pip-licenses>=1.7.1"],
    },
    package_dir={'': source_path},
    zip_safe=False,
    include_package_data=True,
    classifiers=[
        'Development Status :: 0 - PreAlpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Framework :: Framework Independent',
    ],
)
