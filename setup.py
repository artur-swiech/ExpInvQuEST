from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ExpInvQuEST",
    version="0.1.0",
    author="Artur Swiech",
    author_email="artur.swiech@gmail.com",
    description="Inverse QuEST procedures for matrices with exponential autocorrelations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/artur-swiech/ExpInvQuEST",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
