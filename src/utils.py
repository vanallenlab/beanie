from tqdm.auto import tqdm

import random
import multiprocessing

import pandas as pd
import numpy as np
from scipy.stats import zscore

from rpy2.robjects.packages import importr


def CPMNormalisationLogScaling(counts,t1_cells,t2_cells,**kwargs):
    """Function for CPM normalisation of the counts matrix, followed by log-scaling. Also removes genes which are expressed in less that 1% of cells.
    
    Parameters:
        counts                                        counts matrix, without normalisation (genes x cells)
    
    """
    # filter genes that are expressed in less that 1% of the cells
    pct_cells = 0.005    
    counts_filtered = counts.loc[counts.gt(0).sum(axis=1)>int(pct_cells*counts.shape[1]),:]

    df = np.log((counts_filtered*pow(10,6)/counts_filtered.sum(axis=0)) + 1 )
    
    return df
    
    
def SignatureScoringZNormalisedHelper(counts, signature):
    """Helper function for SignatureScoringZNormalised()
    
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
    """Function to do scoring and normalising by null distribution of randomly generated signature of same size.
    
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
    """Function for scoring using mean expression of genes in the gene signature.  
    
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
    """Function to interface with msigdb (http://www.gsea-msigdb.org/gsea/msigdb/genesets.jsp) 
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


