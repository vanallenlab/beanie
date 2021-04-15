import pandas as pd
import numpy as np

from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from .utils import CalculateLog2FoldChange

def FindTopGenes(coeff_matrix, no_of_genes:int):
    correction_method = "bonferroni"
    if no_of_genes>len(coeff_matrix):
        no_of_genes = len(coeff_matrix)
    df = pd.DataFrame(coeff_matrix, columns=["gene_name","log2fold","std_error","direction","robustness_ratio"])
    df = df.set_index("gene_name")
    
    # select only significant genes
    df = df[(df.log2fold>=0.5) & (df.robustness_ratio>=0.9)]
    gr = df.groupby("direction").groups
    try:
        df = df.loc[gr[True],].sort_values(["robustness_ratio","log2fold"],ascending=(False,False))
        return df.iloc[:min(no_of_genes,df.shape[0]),:]
    except KeyError:
        # case when there is no gene in the signature enriched in the direction of interest...
        return None


def FindDriverGenes(signature_name, signature_matrix, counts_matrix, signature_genes, d1, d2, no_of_genes=10):
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
            
    top_genes = FindTopGenes(results_list, no_of_genes)
    return top_genes


def GenerateHeatmapFigure(df1_list, df2_list, signature_list):
    """Function for plotting the actual heatmap.
    
    Parameters:
        df1_list                                      dictionary mapping signature name to df containing mean expression per patient for the gene, for treatment group 1
        df2_list                                      dictionary mapping signature name to df containing mean expression per patient for the gene, for treatment group 2
        signature_list                                list of signature names to plot
    
    """
    v_high = round(max(pd.concat(df1_list.values()).max().max(),pd.concat(df2_list.values()).max().max()))
    v_low = round(min(pd.concat(df1_list.values()).min().min(),pd.concat(df2_list.values()).min().min()))
    
    with sns.plotting_context("notebook", rc={'axes.titlesize' : 12,
                                           'axes.labelsize' : 8,
                                           'xtick.labelsize' : 10,
                                           'ytick.labelsize' : 8}):

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
                        yticklabels = df.index)
                ax.set_ylabel(signature_list[int(i/2)])
            else:
                df = df2_list[signature_list[int(i/2)]]
                im = sns.heatmap(df, ax=ax,
                        cbar=i == 0,
                        vmin=v_low, vmax=v_high,
                        cbar_ax=None if i else cbar_ax, 
                        xticklabels = False if i!=max_i-1 else df2_list[signature_list[0]].columns, 
                        yticklabels = False)

        fig.suptitle("Heatmap for Robust Signatures", fontsize=15)
        plt.subplots_adjust(hspace = 0.05, wspace = 0.01, left=0, right=0.9, top=0.95, bottom=0)
        return

def GenerateHeatmap(counts_matrix, t1_ids, t2_ids, d1, d2, top_genes:dict, signature_list:list):
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
    for sig_name in signature_list:
        
        gene_names = top_genes[sig_name].index.to_list()
        
        df1 = pd.DataFrame(columns=t1_ids)
        for t1 in t1_ids:
            df1[t1] = counts_matrix.loc[d1[t1],gene_names].mean(axis=0)
        df1_dict[sig_name] = df1
        
        df2 = pd.DataFrame(columns=t2_ids)
        for t2 in t2_ids:
            df2[t2] = counts_matrix.loc[d2[t2],gene_names].mean(axis=0)
        df2_dict[sig_name]=df2
        
    return GenerateHeatmapFigure(df1_dict, df2_dict, signature_list)
