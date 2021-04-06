import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def FindTopGenes(coeff_matrix, no_of_genes:int):
    correction_method = "bonferroni"
    if no_of_genes>len(coeff_matrix):
#         print("The signature has lesser number of genes")
        no_of_genes = len(coeff_matrix)
    df = pd.DataFrame(coeff_matrix, columns=["gene_name","coeff1","pval1","coeff2","pval2","log2fold","direction"])
    df = df.set_index("gene_name")
    df.insert(3,"corrected_pval1",multipletests(df.pval1, method = correction_method)[1])
    df.insert(6,"corrected_pval2",multipletests(df.pval2, method = correction_method)[1])

    max_coeff_list = []
    for count in range(0,df.shape[0]):
        if (df.coeff1[count]>df.coeff2[count]) & (df.direction[count]):
            max_coeff_list.append(df.coeff1[count])
        elif (df.coeff1[count]<df.coeff2[count]) & (df.direction[count]==False):
            max_coeff_list.append(df.coeff2[count])
        else:
            max_coeff_list.append(None)
    df["max_coeff"] = max_coeff_list
    
    # select only significant genes
    df = df[(df.corrected_pval1<=0.05) & (df.corrected_pval2<=0.05)]
    df = df.sort_values(by=["log2fold","max_coeff"],ascending=False)
    return df.iloc[:min(no_of_genes,df.shape[0]),:]


def FindDriverGenes(signature_name, signature_matrix, counts_matrix, signature_genes, t1_cells, t2_cells, coeff_name="spearman",no_of_genes=10):
    """ Perform correlation test to find genes which are correlated to the signature score.
    
    Parameters:
        signature_name             name of the signature
        signature_matrix           cells x signatures matrix for calculated signature scores
        counts_matrix              cells x genes counts matrix
        signature_genes            list of genes which are contained in the signature_name
        coeff_name                 type of correlation to use, default = spearman
        no_of_genes                number of top genes required to calculate
    
    """
    if not signature_matrix.index.equals(counts_matrix.index):
        pass
    
    if coeff_name=="spearman":
        test_name = spearmanr
    if coeff_name=="pearson":
        test_name = pearsonr
    if coeff_name=="kendall":
        test_name = kendalltau
        
    x1 = signature_matrix.loc[t1_cells,signature_name].values
    x2 = signature_matrix.loc[t2_cells,signature_name].values
    
    test_coeff = []
    
    for gene in signature_genes:
        if gene in counts_matrix.columns:
            y1 = counts_matrix.loc[t1_cells,gene].values
            y2 = counts_matrix.loc[t2_cells,gene].values
            test1 = test_name(x1,y1)
            test2 = test_name(x2,y2)
            direction = counts_matrix.loc[t1_cells,gene].mean() > counts_matrix.loc[t2_cells,gene].mean()
            log2fold = np.log2(abs(counts_matrix.loc[t1_cells,gene].mean() - counts_matrix.loc[t2_cells,gene].mean()))
            test_coeff.append([gene,test1[0],test1[1],test2[0],test2[1],log2fold,direction])
            
    top_genes = FindTopGenes(test_coeff, no_of_genes)
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
