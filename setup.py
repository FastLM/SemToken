"""
Setup script for SemToken package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().strip().split("\n")

setup(
    name="semtoken",
    version="1.0.0",
    author="Dong Liu, Yanxuan Yu",
    author_email="dong.liu.dl2367@yale.edu, yy3523@columbia.edu",
    description="Semantic-Aware Tokenization for Efficient Long-Context Language Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dongliu/SemToken",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "benchmark": [
            "wandb>=0.12.0",
            "jupyter>=1.0.0",
            "ipywidgets>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "semtoken-benchmark=evaluation.benchmark:main",
            "semtoken-demo=examples.basic_usage:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
