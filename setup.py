"""
Setup script for the Multilayer Citation Model package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    """Read README.md for long description."""
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "A Python package for multilayer citation network modeling and prediction."

# Read requirements
def read_requirements():
    """Read requirements.txt for dependencies."""
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "numpy>=1.19.0",
            "scipy>=1.6.0",
            "pandas>=1.2.0",
            "scikit-learn>=0.24.0",
        ]

setup(
    name="multilayer-citation-model",
    version="1.0.0",
    author="Sadamori Kojaku et al.",
    author_email="",
    description="A Python package for multilayer citation network modeling and prediction",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/skojaku/Legal-Citations",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "examples": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "multilayer-citation-model=multilayer_citation_model.examples:main",
        ],
    },
    include_package_data=True,
    package_data={
        "multilayer_citation_model": ["docs/*.md", "docs/*.pdf"],
    },
    keywords="citation networks, multilayer networks, preferential attachment, machine learning, network science",
    project_urls={
        "Documentation": "https://github.com/skojaku/Legal-Citations/tree/main/libs/authorship_citation_model",
        "Source": "https://github.com/skojaku/Legal-Citations",
        "Tracker": "https://github.com/skojaku/Legal-Citations/issues",
    },
)