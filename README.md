# BEANIE: group Biology EstimAtion iN sIngle cEll

BEANIE is a python package for differential expression analysis in single cell RNA-seq data. It estimates group-level biology between two treatment groups by identifying statistically robust gene signatures. It uses a subsampling based approach to account for over-representation of biology due to differing numbers of cells per sample, and hence helps to reduce the effects of sample biases in differential expression analysis. 

BEANIE interfaces with MsigDb to allow for easy exploratory analysis with experimentally and computationally curated gene sets. It also has inbuilt scoring functions which can be used for scoring these gene sets. In addition to its use for exploratory analysis, BEANIE can also be used to check for robustness of custom signatures obtained from analysis pipelines like scanpy and seurat.

BEANIE also identifies driver genes for each signature/pathway to help characterise the cause of robustness better. It provides a number of modules for visualisation of robustness, driver genes and limitations of sample size.

## Getting Started

To install via pip:

```
pip install beanie
```

To install via github:

```
pip install git+ssh://git@github.com:sjohri20/beanie.git
```

## Software Requirements

python v3.6
