#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages


# requirements
INSTALL_REQUIRES_REPLACEMENTS = {
    'git+git://github.com/ethereum/viper.git@master#egg=viper': 'viper',
}
INSTALL_REQUIRES = list()
with open('requirements.txt') as requirements_file:
    for requirement in requirements_file:
        dependency = INSTALL_REQUIRES_REPLACEMENTS.get(
            requirement.strip(),
            requirement.strip(),
        )

        INSTALL_REQUIRES.append(dependency)

INSTALL_REQUIRES = list(set(INSTALL_REQUIRES))

DEPENDENCY_LINKS = []
if os.environ.get("USE_PYETHEREUM_DEVELOP"):
    # Force installation of specific commits of pyethereum.
    pyethereum_ref = 'ed66e318b8dc34af925096846a77133b201d64b4'   # Oct 22, 2017
    DEPENDENCY_LINKS = [
        'http://github.com/ethereum/pyethereum/tarball/%s#egg=ethereum-9.99.9' % pyethereum_ref
    ]
elif os.environ.get("USE_PYETHEREUM_SIM"):
    # Force installation of specific commits of pyethereum.
    pyethereum_ref = '9875abfc579863018406f2ff5c476f4fc617f92a'
    DEPENDENCY_LINKS = [
        'http://github.com/hwwhww/pyethereum/tarball/sim/%s#egg=ethereum-9.99.9' % pyethereum_ref
    ]

# Force installation of specific commits of viper.
viper_ref = '9a6f972ba459f66e63adcfe9a4ad1c7d2f6ec47a'  # Oct 23, 2017
DEPENDENCY_LINKS.append('http://github.com/ethereum/viper/tarball/%s#egg=viper-9.99.9' % viper_ref)

# *IMPORTANT*: Don't manually change the version here. Use the 'bumpversion' utility.
# see: https://github.com/ethereum/pyethapp/wiki/Development:-Versions-and-Releases
version = '0.0.1'

setup(
    name='sharding',
    version=version,
    description='Ethereum Sharding PoC utilities',
    url='https://github.com/ethereum/sharding',
    packages=find_packages(exclude=('simulation', 'docs')),
    package_data={},
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=INSTALL_REQUIRES,
    dependency_links=DEPENDENCY_LINKS
)
