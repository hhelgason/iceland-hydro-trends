from setuptools import setup, find_packages

setup(
    name="trend_analysis",
    version="0.1.0",
    description="A Python module for analyzing streamflow trends and plotting results",
    author="Hordur Helgason",
    author_email="helgason@uw.edu",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "statsmodels",
        "pymannkendall",
        "geopandas"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
