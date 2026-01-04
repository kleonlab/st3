from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sedd-rnaseq",
    version="0.1.0",
    author="SEDD RNA-seq",
    description="Score-Entropy Discrete Diffusion for masked RNA-seq prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kleonlab/st3",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "full": [
            "matplotlib>=3.4.0",
            "pandas>=1.3.0",
            "scipy>=1.7.0",
            "h5py>=3.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
