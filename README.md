# BEANIE: group Biology EstimAtion iN sIngle cEll

BEANIE is a python package for differential expression analysis in single cell RNA-seq data. It estimates group-level biology between two treatment groups by identifying statistically significant and robust gene signatures. It uses a subsampling based approach to remove the effect of sample biases due to differing numbers of cells per sample, and sample-exclusion based approach  in to quantify robustness in differential signature enrichment analysis. 

Tutorials can be found on the project wiki: https://www.github.com/sjohri20/beanie/wiki

## Installation

To install via github:

```
pip install git+https://github.com/sjohri20/beanie.git
```

## Software Requirements

python v3.6 and above (<= 3.8)
