from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name="BEANIE",
    version="1.0.0",
    description="Bioinformatics tool for differential enrichment of gene signatures in single-cell datasets.",
    author="Shreya Johri",
    author_email="sjohri@g.harvard.edu",
    url="https://github.com/vanallenlab/beanie",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2.0",
        "Operating System :: OS Independent",
    ],
)
