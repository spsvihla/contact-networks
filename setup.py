from setuptools import setup, find_packages

PKG = "contact_networks"

setup(
    name=PKG,
    version="0.1.0",
    description="A Python package for analyzing contact networks and simulating infections thereon",
    author="Sean Svihla",
    packages=find_packages(),  # Automatically find packages in the project
    install_requires=[
        "numpy"
    ],
    python_requires=">=3.6",
)