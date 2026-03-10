from setuptools import setup, find_packages

setup(
    name="renuity-ml-features",
    version="1.0.3",
    description="Shared feature engineering library for Renuity ML",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "ziptimezone",
    ],
    python_requires=">=3.8",
)
