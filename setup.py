from setuptools import setup, find_packages

setup(
    name="pytrance",
    version="0.1.0",
    description="pyTrance: utilities for subcellular spatial transcriptomics analysis",
    author="",
    author_email="",
    packages=find_packages(include=["pytrance", "pytrance.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "tqdm",
        "anndata",
        "scanpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
