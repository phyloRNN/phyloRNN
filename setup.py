import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="phyloRNN",
    version="0.42",
    author="Daniele Silvestro, Thibault Latrille",
    author_email="davide.silvestro@unil.ch, thibault.latrille@unil.ch",
    description="A project for phylogenetic analysis using RNNs",
    url="https://github.com/phyloRNN/phyloRNN",
    packages=setuptools.find_packages(),
    python_requires='>=3.9',
    install_requires=requirements,
)