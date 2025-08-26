#!/usr/bin/env python
"""Setup script for PPG Glucose Estimation."""

from setuptools import setup, find_packages

setup(
    name="ppg-glucose",
    version="1.0.0",
    description="Non-invasive blood glucose estimation from PPG signals",
    author="Amr Hym",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "pyyaml>=6.0",
        "tabulate>=0.9.0",
    ],
    entry_points={
        "console_scripts": [
            "ppg=cli.main:app",
        ],
    },
)