"""
setup.py — House Price Prediction Package
Install with: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [l.strip() for l in f if l.strip() and not l.startswith("#")]

setup(
    name="house-price-prediction",
    version="2.0.0",
    author="Data Science Portfolio",
    description="Production-grade ML pipeline for house price prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/house-price-prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "flake8", "mypy"],
        "shap": ["shap>=0.44"],
    },
    entry_points={
        "console_scripts": [
            "house-predict=predict:main",
            "house-train=model_training:main",
            "house-pipeline=run_pipeline:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="machine-learning, house-price, xgboost, flask, data-science",
)
