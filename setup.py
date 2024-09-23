from setuptools import setup, find_packages

setup(
    name="lgbm-to-code",  # Your package name
    version="0.2",  # Package version
    author="Daniel Gaskins",
    author_email="danielgaskins99@gmail.com",
    description="Convert a trained LGBM instance into conditionals that return the same output as a predict function. Supports javascript, python and C++.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/danielgaskins/lgbm-to-code",  # Your package repository URL
    packages=find_packages(),  # Automatically finds and includes your package modules
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "lightgbm"  # LightGBM dependency
    ]
)