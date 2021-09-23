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


#### Gene Importance Ranking

Beanie can rank genes by their importance in calculation of score. This step is particularly useful for users when they have very large number of genes in a signature and want to know relative importance of genes.

```
bobj.DriverGenes()
bobj.GetDriverGenesSummary()
```

#### Visualisation

Beanie has many functions to aid the user to visualise the results.

Plot 1: Bar Plot
All signatures which have significant p values are shown in this plot, with the height of the bar being the log(p). Robustness and directionality are also shown in the plot.

```
beanie_obj_melanoma.BarPlot()
```

Plot 2: Dropout Plot
This plot is specially for visualising the non-robust signatures in detail. It shows which patients led to the non-robustness of a particular signature, and also how many signatures became non-robust because of one particular patient. This is important to find patient outliers whose biology is very different from the other samples in the dataset.

```
beanie_obj_melanoma.PatientDropoutPlot()
```

Plot 3: Heatmap
This plot shows the top genes for each signature. By default it plots for 5 signatures which have lowest corrected p-values, but it is recommended that the user uses this plot to visualise for other signatures too. A list of signature names to be plotted can be passed as a parameter to this function.

```
bobj.HeatmapDriverGenes(signature_names = ["signature_1", "signature_2", "signature_3", "signature_4", "signature_5"])
```

Plot 4: Upset Plots
This plot is used for visualising the overlap between the top 10 ranked genes across different signatures. A list of signature names can be provided. If no list is provided, by default, the top 5 signatures with lowest p values are used for plotting.

```
bobj.UpsetPlotDriverGenes(signature_names=["signature_1", "signature_2", "signature_3", "signature_4", "signature_5"])
```

Similarly, overlap between the all genes for signatures can also be seen. A list of signature names can be provided. If no list is provided, by default, the top 5 signatures with lowest p values are used for plotting.

```
bobj.UpsetPlotSignatureGenes(signature_names=["signature_1", "signature_2", "signature_3", "signature_4", "signature_5"])
```

