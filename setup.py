import setuptools
from setuptools import setup

# name is the package name, tweeter in my example
# version is the package version
# description is a short summary of the package's purpose
# author is of course the package's author
# packages is the name of your package
# install_requires is a list of dependencies for the package, much like "dependencies" in package.json
with open("README.md") as f:
    long_description = f.read()


setup(
    name="ai-toolkit",
    version="0.0.1",
    author="Tyler Yep",
    author_email="tyep10@gmail.com",
    description="AI Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tyleryep/ai-toolkit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
