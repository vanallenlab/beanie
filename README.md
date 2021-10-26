# BEANIE: group Biology EstimAtion iN sIngle cEll

BEANIE is a python package for differential enrichment analysis in single cell RNA-seq data. It uses Monte-Carlo approximations to conclude statistical significance and further contextualises the statistical significance with a null distribution to identify possible biologically relevant gene signatures. It additionally ensures that the final statistically significant gene signatures are not confounded by patient-specific biology via employing a leave-one-out cross-validation approach to estimate the robustness of these statistically significant gene signatures to sample exclusion.

It decreases the false positive rate by more than 10-fold as compared to the conventional methods such as Mann-Whitney U test and Generalised Linear Models.

![](https://github.com/sjohri20/beanie/blob/main/figs/false_positive.png)

Tutorials can be found on the project [wiki](https://www.github.com/sjohri20/beanie/wiki)

## Installation

To install via github:

```
pip install git+https://github.com/sjohri20/beanie.git
```

## Software Requirements

python v3.6 and above (<= 3.8)


## Citation

If you use our package, please cite the preprint: Johri S., Bi K., Fu J., Titchen B., Conway J., Crowdis J., Vokes N., Fan Z, Fong L., Park J., Liu D., He MX., Van Allen E. (2021) Dissecting tumor cellprograms through group biology estimation in clinical single cell transcriptomics. biorxiv.