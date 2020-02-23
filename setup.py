from setuptools import setup

# name is the package name, tweeter in my example
# version is the package version
# description is a short summary of the package's purpose
# author is of course the package's author
# packages is the name of your package
# install_requires is a list of dependencies for the package, much like "dependencies" in package.json

setup(
    name="ml-toolkit",
    version="0.0.1",
    description="ML Toolkit",
    author="Tyler Yep",
    packages=["tweeter"],
    install_requires=["tweepy>=3.8.0"],
)