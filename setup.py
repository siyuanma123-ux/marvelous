from setuptools import setup, find_packages

setup(
    name="virtual_skin",
    version="0.1.0",
    description=(
        "Omics-Constrained Multi-Scale Virtual Skin for Transdermal Drug Delivery, "
        "Screening, and Target Prediction"
    ),
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "torch>=2.0.0",
        "scanpy>=1.9.0",
        "anndata>=0.10.0",
        "deepxde>=1.10.0",
        "pyro-ppl>=1.8.0",
    ],
)
