from setuptools import setup


with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="rba",
    version="0.0.1",
    description="Code for *The Rebalancing Act: A Geographical Community-Based Approach to Computational Gerrymandering Analysis*",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gcrsef-gang/the-rebalancing-act",
    install_requires=[
        "matplotlib==3.5.1", 
        "pandas==1.4.1",
        "gerrychain==0.2.20",
        "maup==1.0.8",
        "welford==0.2.5"
    ]
)