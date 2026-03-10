from setuptools import setup, find_packages

setup(
    name="renuity-ml-features",
    version="1.0.4",  # bumped version
    description="Shared feature engineering library for Renuity ML",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[], # Empty now!
    python_requires=">=3.8",
)
