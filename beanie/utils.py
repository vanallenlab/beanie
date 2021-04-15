from tqdm.auto import tqdm

import random
import multiprocessing

import pandas as pd
import numpy as np
from scipy.stats import zscore,sem

from rpy2.robjects.packages import importr


def CPMNormalisationLogScaling(counts,**kwargs):
    """
    Function for CPM normalisation of the counts matrix, followed by log-scaling. Also removes genes which are expressed in less that 1% of cells.
    
    Parameters:
        counts                                        counts matrix, without normalisation (genes x cells)
    
    """
    # filter genes that are expressed in less that 1% of the cells
    pct_cells = 0.005    
    counts_filtered = counts.loc[counts.gt(0).sum(axis=1)>int(pct_cells*counts.shape[1]),:]

    df = np.log((counts_filtered*pow(10,6)/counts_filtered.sum(axis=0)) + 1 )
    
    return df
    
    
def SignatureScoringZNormalisedHelper(counts, signature):
    """
    Helper function for SignatureScoringZNormalised()
    
    """
    random.seed(3120)

    score_df = pd.DataFrame(index=counts.columns)
    gene_list = counts.index.to_list()
    for x in signature.columns:
        genes = signature[x].dropna().to_list()
        temp = set(genes).intersection(gene_list)
#         if len(temp)<len(genes):
#             print("Dropping genes",set(genes).difference(temp),"from gene signature", x)
            
        # permutation 200 times    
        null_dist = pd.DataFrame(index=counts.columns)
        for i in range(200):
            temp = random.sample(gene_list,len(temp))
            null_dist[i] = counts.loc[temp,:].mean(axis=0)
        score_df[x] = (counts.loc[temp,:].mean(axis=0) - null_dist.mean(axis=1))/null_dist.var(axis=1)
    return(score_df)


def SignatureScoringZNormalised(counts, signatures,n_cores=4):
    """
    Function to do scoring and normalising by null distribution of randomly generated signature of same size.
    
    Parameters:
        counts                                      counts matrix, without normalisation
        signatures                                  signature score matrix
        n_cores                                     number of cpu cores on which to parallelise
    
    """
    df_split = np.array_split(signatures, n_cores, axis=1)
    items = [(counts,d) for d in df_split]
    with multiprocessing.Pool(processes=n_cores) as pool:
        df = pd.concat(pool.starmap(SignatureScoringZNormalisedHelper, items), axis=1)
    return df
    

def SignatureScoringMean(counts, signatures):
    """
    Function for scoring using mean expression of genes in the gene signature.  
    
    Parameters:
        counts                                      counts matrix, without normalisation
        signatures                                  signature score matrix
    
    """
    score_df = pd.DataFrame(index=counts.columns)
    gene_list = counts.index.to_list()

    for x in signatures.columns:
        genes = signatures[x].dropna().to_list()
        temp = set(genes).intersection(gene_list)
#         if len(temp)<len(genes):
#             print("Dropping genes",set(genes).difference(temp),"from gene signature", x)
        score_df[x] = counts.loc[temp,:].mean(axis=0)    
    
    return score_df
    
def GetSignaturesMsigDb(msigdb_species,msigdb_category,msigdb_subcategory=None):
    """
    Function to interface with msigdb (http://www.gsea-msigdb.org/gsea/msigdb/genesets.jsp) 
    and extract pathways/signatures. Alternative to providing a user defined signature file.
    Uses R package msigdb for interface.
    thought - can I remove dependence from this package?
    
    Parameters:
        msigdb_species                              species for msigdb
        msigdb_category                             categories: chosen from H,C1,...C8
        msigdb_subcategory                          if present, for eg in case of C2.
    
    """
    print("Extracting signatures from MsigDB: http://www.gsea-msigdb.org/gsea/msigdb/genesets.jsp")
    msig = importr("msigdbr")
    if msigdb_subcategory ==None:
        r_df = msig.msigdbr(species = msigdb_species, category=msigdb_category)
    else:
        r_df = msig.msigdbr(species = msigdb_species, category=msigdb_category, subcategory=msigdb_subcategory)

    pd_df = pd.DataFrame(r_df,index=r_df.colnames).T
    m_df = pd_df[["gs_name","gene_symbol"]]
    temp = m_df.groupby(['gs_name']).groups.keys()
    m_df = pd.DataFrame(list(m_df.groupby('gs_name')["gene_symbol"].unique()), index=m_df.groupby(['gs_name']).groups.keys()).T
    return m_df

def CalculateLog2FoldChangeHelper(a1,a2):
    
    # note: values in log-normalised matrix are all positive
    gr1 = np.log2(np.mean([y for x,y in a1.items()])+0.000001)
    gr2 = np.log2(np.mean([y for x,y in a2.items()])+0.000001)
    log2fold = abs(gr1-gr2)
    direction = gr1>gr2                
    return [log2fold,direction]       
    

def CalculateLog2FoldChange(signature_genes,counts_matrix,d1,d2):
    """
    Function for calculating log2fold change and direction. Helper function for driver_genes module.
    
    Parameters:
        counts_matrix              cells x genes counts matrix
        signature_genes            list of genes which are contained in the signature_name
        d1                         directionary mapping patient id to cells in group of interest
        d2                         directionary mapping patient id to cells in reference group
        
    """
    t1_ids = list(d1.keys())
    t2_ids = list(d2.keys())
    
    test_coeff = []
    for gene in signature_genes:
        if gene in counts_matrix.columns:
            a1=dict()
            a2=dict()
            for id_name in t1_ids:
                a1[id_name] = counts_matrix.loc[d1[id_name],gene].mean()
            for id_name in t2_ids:
                a2[id_name] = counts_matrix.loc[d2[id_name],gene].mean()
            
            # check robustness
            fold_list = []
            for fold in t1_ids:
                a1_temp = {key: value for key, value in a1.items() if key != fold}
                fold_list.append(CalculateLog2FoldChangeHelper(a1_temp,a2))     
            for fold in t2_ids:
                a2_temp = {key: value for key, value in a2.items() if key != fold}
                fold_list.append(CalculateLog2FoldChangeHelper(a1,a2_temp))
            
            log2fold = np.mean([x[0] for x in fold_list])
            std_err = sem([x[0] for x in fold_list])  
            dirs = [x[1] for x in fold_list]
            direction = max(dirs, key = dirs.count)
            robustness_ratio = dirs.count(direction)/len(dirs)
            test_coeff.append([gene,log2fold,std_err,direction,robustness_ratio])    
            
    return test_coeff



def CalculateLog2FoldChangeSigScores(signature_scores,d1,d2):
    """
    
    """
    
    t1_ids = list(d1.keys())
    t2_ids = list(d2.keys())
    
    a1 = pd.DataFrame(index=signature_scores.columns)
    a2 = pd.DataFrame(index=signature_scores.columns)
    for idx in t1_ids:
        a1[idx] = signature_scores.loc[d1[idx],:].mean()
    for idx in t2_ids:
        a2[idx] = signature_scores.loc[d2[idx],:].mean()
    
    fold_log2fold = pd.DataFrame(index=signature_scores.columns)
    fold_direction = pd.DataFrame(index=signature_scores.columns)
    for fold in t1_ids:
        gr1 = np.log2(a1.drop([fold], axis=1).mean(axis=1)+0.000001)
        gr2 = np.log2(a2.mean(axis=1)+0.000001)
        fold_log2fold["excluded_"+fold] = abs(gr1-gr2)
        fold_direction["excluded_"+fold] = gr1>gr2
    for fold in t2_ids:
        gr1 = np.log2(a1.mean(axis=1)+0.000001)
        gr2 = np.log2(a2.drop([fold], axis=1).mean(axis=1)+0.000001)
        fold_log2fold["excluded_"+fold] = abs(gr1-gr2)
        fold_direction["excluded_"+fold] = gr1>gr2

    results_df = pd.DataFrame(index=signature_scores.columns, columns=["log2fold","std_err","direction","robustness_direction"])
    
    results_df["log2fold"] = fold_log2fold.mean(axis=1)
    results_df["std_err"] = fold_log2fold.sem(axis=1)
    results_df["direction"] = fold_direction.mode(axis=1)[0]
    
    tot = fold_direction.shape[1]
    rob_ratios = []
    for i in range(fold_direction.shape[0]):
        if results_df.direction[i]:
            rob_ratios.append(fold_direction.iloc[i,:].astype(int).value_counts()[1]/tot)
        else:
            rob_ratios.append(fold_direction.iloc[i,:].astype(int).value_counts()[0]/tot)
    results_df["robustness_direction"] = rob_ratios
    
    return results_df
        
    