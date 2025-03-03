# BEANIE

## Dissecting tumor cell programs through group biology estimation in clinical single-cell transcriptomics

[Nature Communications Paper](https://www.nature.com/articles/s41467-025-57377-6) | [Analysis](https://github.com/vanallenlab/beanie-analysis) | [AACR '23 Oral Presentation](https://aacrjournals.org/cancerres/article/83/7_Supplement/1120/722439/Abstract-1120-Dissecting-tumor-cell-programs)| [Cite Us](https://github.com/vanallenlab/beanie-analysis#citation)

BEANIE is a non-parameteric method for identification of differentially expressed gene signatures in clinical single-cell RNA-seq datasets. It can compare between two clinical groups of patients that share a subpopulation of cells, and calculates a biologically contextualized p-value (empirical p-value) and robustness for each gene signature. Tutorials can be found on the project [wiki](https://www.github.com/vanallenlab/beanie/wiki).

<!-- It decreases the false positive rate by more than 10-fold as compared to the conventional methods such as Mann-Whitney U test and Generalised Linear Models.

![](https://github.com/vanallenlab/beanie/blob/main/figs/false_positive.png) -->


### Setting up a virtual environment

It is recommended to run BEANIE in a separate conda environment. To setup a new environment -

```
# Create conda environment
conda create --name beanie_env python=3.7

# Activate conda environment
conda activate beanie_env

```

### Installation

BEANIE v1.0.0 can be installed directly via github -

```
pip install git+https://github.com/vanallenlab/beanie.git
```

Requirements: python v3.7 and above.
<!-- - Java v1.8 and above. -->


### Citation
If you've found this work useful, please cite the following :

```
Johri, S., Bi, K., Titchen, B.M. et al. Dissecting tumor cell programs through group biology estimation in clinical single-cell transcriptomics. Nat Commun 16, 2090 (2025). https://doi.org/10.1038/s41467-025-57377-6
```

### Issues
Please report issues directly to sjohri@g.harvard.edu.
