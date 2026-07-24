from setuptools import setup, find_packages

setup(
    name="lgbm-to-code",  # Your package name
    version="0.3.0",
    author="Daniel Gaskins",
    author_email="hello@danielgaskins.com",
    description="Generate dependency-free Python, C++, or JavaScript raw-score inference from trained LightGBM models.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/danielgaskins/lgbm-to-code",  # Your package repository URL
    packages=find_packages(),  # Automatically finds and includes your package modules
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        "lightgbm>=4.0"
    ],
    extras_require={
        "test": ["numpy>=1.23", "pytest>=7"],
    },
)
