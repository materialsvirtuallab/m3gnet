"""
Installation for M3GNet
"""

import os
import re

import numpy as np
from setuptools import find_packages, setup

this_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open("m3gnet/__init__.py", encoding="utf-8") as fd:
    for line in fd.readlines():
        m = re.search('__version__ = "(.*)"', line)
        if m:
            version = m.group(1)
            break

setup(
    name="m3gnet",
    version=version,
    install_requires=[
        "pymatgen",
        "tensorflow",
        "numpy",
        "monty",
        "sympy",
        "ase",
    ],
    description="Materials Graph with Three-body Interactions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chi Chen",
    author_email="chen08013@gmail.com",
    download_url="https://github.com/materialsvirtuallab/m3gnet",
    license="BSD",
    extras_require={
        "model_saving": ["h5py"],
        "tensorflow with gpu": ["tensorflow-gpu>=2.7"],
    },
    packages=find_packages(),
    package_data={
        "m3gnet": ["*.json", "*.md"],
    },
    include_package_data=True,
    include_dirs=np.get_include(),
    keywords=[
        "materials",
        "science",
        "machine",
        "learning",
        "deep",
        "graph",
        "networks",
        "neural",
        "force field",
        "interatomic potential",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "m3g = m3gnet.cli:main",
        ]
    },
)
