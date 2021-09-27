import pandas as pd
import numpy as np

from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from .utils import CalculateLog2FoldChange, OutlierDetector

def FindTopGenes(coeff_matrix):
#     correction_method = "bonferroni"
    
#     if no_of_genes>len(coeff_matrix):
#         no_of_genes = len(coeff_matrix)
    df = pd.DataFrame(coeff_matrix, columns=["gene_name","gr1_outlier","gr2_outlier",
                                             "log2fold","log2fold_outlier","std_error","direction","robustness_ratio"])
    df = df.set_index("gene_name")

    # select only significant genes
    df = df[(df.log2fold>=0.5) & (df.robustness_ratio>=0.9)]
    gr = df.groupby("direction").groups
    try:
        df = df.loc[gr[True],].sort_values(["log2fold_outlier", "gr1_outlier", "gr2_outlier", 
                                            "log2fold", "std_error", "robustness_ratio"],
                                           ascending=(True, True, True, False, True, False))
        return df
    except KeyError:
        # case when there is no gene in the signature enriched in the direction of interest...
        return None


def FindDriverGenes(signature_name, signature_matrix, counts_matrix, signature_genes, d1, d2):
    """ Perform correlation test to find genes which are correlated to the signature score.
    
    Parameters:
        signature_name             name of the signature
        signature_matrix           cells x signatures matrix for calculated signature scores
        counts_matrix              cells x genes counts matrix
        signature_genes            list of genes which are contained in the signature_name
        d1                         directionary mapping patient id to cells in group of interest
        d2                         directionary mapping patient id to cells in reference group
        no_of_genes                number of top genes required to calculate
    
    """
    if not signature_matrix.index.equals(counts_matrix.index):
        pass

    results_list = CalculateLog2FoldChange(signature_genes,counts_matrix,d1,d2)       
    top_genes = FindTopGenes(results_list)
    return top_genes


def GenerateHeatmapFigure(df1_list, df2_list, signature_list, **kwargs):
    """Function for plotting the actual heatmap.
    
    Parameters:
        df1_list                                      dictionary mapping signature name to df containing mean expression per patient for the gene, for treatment group 1
        df2_list                                      dictionary mapping signature name to df containing mean expression per patient for the gene, for treatment group 2
        signature_list                                list of signature names to plot
    
    """
    # how to place the labels of the heatmap
    orientation = "horizontal"
    cmap_name = "Blues"
    
    v_high = max(pd.concat(df1_list.values()).max().max(),pd.concat(df2_list.values()).max().max())
    v_low = min(pd.concat(df1_list.values()).min().min(),pd.concat(df2_list.values()).min().min())
    with sns.plotting_context("notebook", rc={'axes.titlesize' : 12,
                                           'axes.labelsize' : 8,
                                           'xtick.labelsize' : 10,
                                           'ytick.labelsize' : 8,
                                           'font.name' : 'Arial'}):

        fig, axs = plt.subplots(len(signature_list),2,figsize=(10, max(1.4*max(len(signature_list),5), 0.5*df1_list[signature_list[0]].shape[0])), dpi=300)
        cbar_ax = fig.add_axes([0.91,.3,.03,.4])
        max_i = len(axs.flat)
        for i, ax in enumerate(axs.flat):
            if i%2==0:
                df = df1_list[signature_list[int(i/2)]]
                im = sns.heatmap(df, ax=ax,
                        cbar=i == 0,
                        vmin=v_low, vmax=v_high,
                        cbar_ax=None if i else cbar_ax, 
                        xticklabels = False if i!=max_i-2 else df1_list[signature_list[0]].columns,
                        yticklabels = df.index,
                        cmap = cmap_name)
                if orientation == "horizontal":
                    lab = signature_list[int(i/2)]
                    lab_len = len(signature_list[int(i/2)])
                    ax.set_ylabel(lab[0:int(lab_len/2)]+"\n"+lab[int(lab_len/2):], labelpad=50, rotation=0)
                else:
                    ax.set_ylabel(signature_list[int(i/2)], labelpad=10+10*(i%4))

            else:
                df = df2_list[signature_list[int(i/2)]]
                im = sns.heatmap(df, ax=ax,
                        cbar=i == 0,
                        vmin=v_low, vmax=v_high,
                        cbar_ax=None if i else cbar_ax, 
                        xticklabels = False if i!=max_i-1 else df2_list[signature_list[0]].columns, 
                        yticklabels = False,
                        cmap = cmap_name)

        fig.suptitle("Heatmap for Robust Signatures", fontsize=15)
        plt.subplots_adjust(hspace = 0.05, wspace = 0.01, left=0, right=0.9, top=0.95, bottom=0)
        return

def GenerateHeatmap(counts_matrix, t1_ids, t2_ids, d1, d2, top_genes:dict, signature_list:list, no_genes=10, **kwargs):
    """Function to prepare data for GenerateHeatmapFigure()
    
    Parameters: 
        counts                                       (cells x genes) counts matrix
        t1_ids                                       list of patient ids in treatment group 1
        t2_ids                                       list of patient ids in treatment group 2
        d1                                           dictionary mapping patient ids to cell ids for treatment group 1
        d2                                           dictionary mapping patient ids to cell ids for treatment group 2
        top_genes                                    dictionary mapping signature name to a df of top gene names, p val and correlation coeff
        signature_list                               list of signature names to be plotted in the heatmap
    
    """
    df1_dict = dict()
    df2_dict = dict()
    sig_list_robust = []
    gr1_pats = len(t1_ids)
    gr2_pats = len(t2_ids)
    
    all_genes = list()
    for df in top_genes.values():
        try:
            all_genes.extend(df.index)
        except AttributeError:
            continue

    for sig_name in signature_list:
        try:
            gene_names = top_genes[sig_name].index.to_list()
            gene_names = gene_names[:min(no_genes,len(gene_names))]
            sig_list_robust.append(sig_name)
        except AttributeError:
            print("Signature " + sig_name + " does not have a robust gene differential.")
            continue
        
        df1 = pd.DataFrame(columns=t1_ids)
        for t1 in t1_ids:
            df1[t1] = counts_matrix.loc[d1[t1],gene_names].mean(axis=0)
        df1_dict[sig_name] = df1
        
        df2 = pd.DataFrame(columns=t2_ids)
        for t2 in t2_ids:
            df2[t2] = counts_matrix.loc[d2[t2],gene_names].mean(axis=0)
        df2_dict[sig_name]=df2
        
        # scaling each gene for plot:
        for gene in gene_names:
            temp = np.append(df1_dict[sig_name].loc[gene,:].values, df2_dict[sig_name].loc[gene,:].values)
            temp_scaled = (temp-np.min(temp))/(np.max(temp)-np.min(temp))
            df1_dict[sig_name].loc[gene,:] = temp_scaled[:gr1_pats]
            df2_dict[sig_name].loc[gene,:] = temp_scaled[-gr2_pats:]

    if len(sig_list_robust)==0:
        print("None of the Signatures have robust differential gene expression")
        return
    
    return GenerateHeatmapFigure(df1_dict, df2_dict, sig_list_robust,**kwargs)

