import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="solv-bert", # Replace with your own username
    version="0.0.7",
    author="An Su",
    author_email="ansu@zjut.edu.cn",
    description='''This is the code for "SolvBERT for solvation free energy and solubility prediction: a demonstration of an NLP model for predicting the properties of molecular complexes" paper. The preprint of this paper can be found in ChemRxiv with https://doi.org/10.26434/chemrxiv-2022-0hl5p''',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitee.com/su-zjut/solv-bert",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
