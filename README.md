# BEANIE: group Biology EstimAtion iN sIngle cEll

BEANIE is a python package for identification of differentially-enriched gene signatures in single-cell RNA-seq datasets. It can compare between two clinical groups of patients that share a subpopulation of cells, and calculates a biologically contextualized p-value (empirical p-value) and robustness for each gene signature. Tutorials can be found on the project [wiki](https://www.github.com/vanallenlab/beanie/wiki).

<!-- It decreases the false positive rate by more than 10-fold as compared to the conventional methods such as Mann-Whitney U test and Generalised Linear Models.

![](https://github.com/vanallenlab/beanie/blob/main/figs/false_positive.png) -->


## Setting up a virtual environment

It is recommended to run BEANIE in a separate conda environment. To setup a new environment -

```
# Create conda environment
conda create --name beanie_env python=3.7

# Activate conda environment
conda activate beanie_env

```

## Installation

BEANIE v1.0.0 can be installed directly via github -

```
pip install git+https://github.com/vanallenlab/beanie.git
```

**Requirements**

- python v3.7 and above.
<!-- - Java v1.8 and above. -->


## Citation

If you use our package, please cite the preprint: 


>Johri S., Bi K., Titchen B., Fu J., Conway J., Crowdis J., Vokes N., Fan Z, Fong L., Park J., Liu D., He MX., Van Allen E. (2021) Dissecting tumor cellprograms through group biology estimation in clinical single cell transcriptomics. biorxiv.
