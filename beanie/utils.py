from tqdm.auto import tqdm

import os
from datetime import datetime

import random
import multiprocessing

import pandas as pd
import numpy as np
from scipy.stats import zscore, sem
from math import sqrt

from rpy2.robjects.packages import importr

from .differential_expression import table_mannwhitneyu

def CPMNormalisationLogScaling(counts,**kwargs):
    """
    Function for CPM normalisation of the counts matrix, followed by log-scaling. Also removes genes which are expressed in less that 1% of cells.
    
    Parameters:
        counts                                        counts matrix, without normalisation (genes x cells)
    
    """
    # filter genes that are expressed in less that 1% of the cells
    # pct_cells = 0.005    
    # counts_filtered = counts.loc[counts.gt(0).sum(axis=1)>int(pct_cells*counts.shape[1]),:]

    df = np.log((counts*pow(10,6)/counts.sum(axis=0)) + 1 )
    
    return df
    

def GenerateNullDistributionSignatures(signature,sorted_genes,no_iters=200):    
    """Generate sets of random signatures of variable sizes"""
    
    print("Generating background distribution of random signatures...")
    q20 = np.quantile(sorted_genes,0.20)
    q40 = np.quantile(sorted_genes,0.40)
    q60 = np.quantile(sorted_genes,0.60)
    q80 = np.quantile(sorted_genes,0.80)
    
    size_dict = {}
    for x in signature.columns:
        size = round(len(signature[x].dropna())/5)*5
        if size not in size_dict.keys():
            size_dict[size]=[x]
        else:
            size_dict[size].append(x)
            
    random_set_dict = {}
    for key in size_dict.keys():
        random_set_dict[key] = [random.sample(sorted_genes[sorted_genes<q20].index.to_list(),int(key/5)) + random.sample(sorted_genes[(sorted_genes>q20) & (sorted_genes<q40)].index.to_list(),int(key/5)) + random.sample(sorted_genes[(sorted_genes>q40) & (sorted_genes<q60)].index.to_list(),int(key/5)) + random.sample(sorted_genes[(sorted_genes>q60) & (sorted_genes<q80)].index.to_list(),int(key/5)) + random.sample(sorted_genes[sorted_genes>q80].index.to_list(),int(key/5)) for i in range(no_iters)]
    
    print("Storing temp random sig files in directory...")
    dateTimeObj = datetime.now()
    dir_name = "temp_files_"+dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    os.mkdir(dir_name)
    for key in size_dict.keys():
        pd.DataFrame(random_set_dict[key], index=["random_sig_"+str(i) for i in range(no_iters)]).to_csv("./" + dir_name + "/" + str(key) + ".gmt", sep="\t", header=None)
        
    return random_set_dict,dir_name

def DownSampleCellsForPValCorrection(d:dict, t_cells:list, subsample_per_pat:int , no_subsample_cells:int, subsample_mode="max"):
    
    group = []
    for cells in d.values():
        group.extend(np.random.choice(cells, size=min(subsample_per_pat, len(cells)), replace=False))

    # make up for cells that would have been sampled from excluded sample
    replacements_needed = no_subsample_cells - len(group)
    
    if replacements_needed!=0:
        replacement_group = d
        already_chosen = group
        replacement_candidates = []
        for sample_cells in replacement_group.values():
            replacement_candidates.append(sorted(set(sample_cells)))

        # weight for ~equal representation of samples
        replacement_weights = np.hstack([[1 / len(x)] * len(x) for x in replacement_candidates])
        replacement_weights /= replacement_weights.sum()
        replacement_candidates = np.hstack(replacement_candidates)
        if subsample_mode=="max":
            already_chosen.extend(np.random.choice(replacement_candidates,
                                               size=replacements_needed,
                                               replace=False,
                                               p=replacement_weights))
        else:
            already_chosen.extend(np.random.choice(replacement_candidates,
                                               size=replacements_needed,
                                               replace=False))

    return group
    
def PValCorrectionPermutationTest(expression, t1_cells, t2_cells,  
                                  U_uncorrected: pd.Series, p_uncorrected: pd.Series, alternative="two-sided"):
    """Find the corrected pval based on mann whitney test.
    
    Parameters:
        expression                             (cells x signatures) matrix
        t1_cells
        t2_cells
    
    """
    U, p = table_mannwhitneyu(expression.T, t1_cells, t2_cells, alternative)
    p_corr_U = min(len(U[U>U_uncorrected])/len(U),len(U[U<U_uncorrected])/len(U))
    p_corr_p = len(p[p<p_uncorrected])/len(p)
    return p_corr_U, p_corr_p

        
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


def SignatureScoringCombinedZScore(counts, signatures):
    """
    Function for scoring using mean z-score of genes in the gene signature. Will automatically drop genes which are not present in the counts matrix.
    
    Parameters:
        counts                                      counts matrix, without normalisation
        signatures                                  signature score matrix
    
    """
    score_df = pd.DataFrame(index=counts.columns)
    gene_list = counts.index.to_list()
    counts_scaled = pd.DataFrame(zscore(counts,axis=1), index=counts.index, columns=counts.columns)
    
    for x in signatures.columns:
        genes = signatures[x].dropna().to_list()
        temp = set(genes).intersection(gene_list)
        score_df[x] = counts_scaled.loc[temp,:].mean(axis=0)/sqrt(len(temp)) 
    
    return score_df

def SignatureScoringMean(counts, signatures):
    """
    Function for scoring using mean expression of genes in the gene signature. Will automatically drop genes which are not present in the counts matrix.
    
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

def OutlierDetector(a:dict):
    a_vals = list(a.values())
    q25 = np.quantile(a_vals,0.25)
    q75 = np.quantile(a_vals,0.75)
    outlier_bottom_lim = q25 - 1.5 * (q75 - q25)
    outlier_upper_lim = q75 + 1.5 * (q75 - q25)
    if (a_vals>outlier_upper_lim).sum()>0 or (a_vals<outlier_bottom_lim).sum()>0:
        return True
    else:
        return False

def OutlierDetectorFold(a:list):
    q25 = np.quantile(a,0.25)
    q75 = np.quantile(a,0.75)
    outlier_bottom_lim = q25 - 1.5 * (q75 - q25)
    outlier_upper_lim = q75 + 1.5 * (q75 - q25)
    if (a>outlier_upper_lim).sum()>0 or (a<outlier_bottom_lim).sum()>0:
        return True
    else:
        return False
    
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
            
            # find outliers
            gr1_outlier = OutlierDetector(a1)
            gr2_outlier = OutlierDetector(a2)
            
            # check robustness
            fold_list = []
            for fold in t1_ids:
                a1_temp = {key: value for key, value in a1.items() if key != fold}
                fold_list.append(CalculateLog2FoldChangeHelper(a1_temp,a2)) 
            for fold in t2_ids:
                a2_temp = {key: value for key, value in a2.items() if key != fold}
                fold_list.append(CalculateLog2FoldChangeHelper(a1,a2_temp))
            
            log2fold = np.mean([x[0] for x in fold_list])
            log2fold_outlier = OutlierDetectorFold([x[0] for x in fold_list])
            std_err = sem([x[0] for x in fold_list])
            dirs = [x[1] for x in fold_list]
            direction = max(dirs, key = dirs.count)
            robustness_ratio = dirs.count(direction)/len(dirs)
            test_coeff.append([gene, gr1_outlier, gr2_outlier, log2fold, log2fold_outlier,
                               std_err, direction, robustness_ratio])    
            
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


def CalculateSubsampleSize(metad, treatment_group_names, subsample_mode, **kwargs):
    
    def choose_cells(group1_cells, group2_cells, group1_sample_cells, group2_sample_cells, 
             excluded_cells, excluded, subsample_mode, excluded_from_group1=True):
        """Select cells to include in each iteration."""
        # choose cells
        group1 = []
        group2 = []
        cells_chosen = []

        for cells in group1_cells.values():
            group1.extend(np.random.choice(cells, size=min(group1_sample_cells, len(cells)), replace=False))
        for cells in group2_cells.values():
            group2.extend(np.random.choice(cells, size=min(group2_sample_cells, len(cells)), replace=False))

        # make up for cells that would have been sampled from excluded sample
        replacements_needed = min(group1_sample_cells if excluded_from_group1 else group2_sample_cells,
                                  excluded_cells)
        replacement_group = group1_cells if excluded_from_group1 else group2_cells
        already_chosen = group1 if excluded_from_group1 else group2
        replacement_candidates = []
        for sample_cells in replacement_group.values():
            sample_candidates = set(sample_cells) - set(already_chosen)
            if sample_candidates:
                replacement_candidates.append(sorted(sample_candidates))

        # weight for ~equal representation of samples
        replacement_weights = np.hstack([[1 / len(x)] * len(x) for x in replacement_candidates])
        replacement_weights /= replacement_weights.sum()
        replacement_candidates = np.hstack(replacement_candidates)
        if subsample_mode =="max":
            already_chosen.extend(np.random.choice(replacement_candidates,
                                               size=replacements_needed,
                                               replace=False,
                                               p=replacement_weights))
        else:
            already_chosen.extend(np.random.choice(replacement_candidates,
                                               size=replacements_needed,
                                               replace=False))
            
            
    t1_ids = list(set(metad[metad.treatment_group==treatment_group_names[0]].patient_id))
    t2_ids = list(set(metad[metad.treatment_group==treatment_group_names[1]].patient_id))

    d1_all = {}
    for xid in t1_ids:
        d1_all[xid]=metad[metad.patient_id==xid]["patient_id"].index.to_list()

    d2_all = {}
    for xid in t2_ids:
        d2_all[xid]=metad[metad.patient_id==xid]["patient_id"].index.to_list()

    
    min_size = int(5*round(metad.patient_id.value_counts()[-1]/5))
    max_size = int(5*round(metad.patient_id.value_counts()[0]/5))
    steps = 5
    
    exclusions = []
    if len(d1_all) > 1:
        exclusions.extend(list(d1_all.keys()))
    if len(d2_all) > 1:
        exclusions.extend(list(d2_all.keys()))
        
    for i in range(min_size,max_size,steps):
        for excluded in sorted(exclusions):
            try: 
                size = choose_cells(
                    {x: y for x, y in d1_all.items() if x != excluded}, 
                    {x: y for x, y in d2_all.items() if x != excluded}, 
                    i, i, len({**d1_all, **d2_all}[excluded]), 
                    excluded, subsample_mode, excluded in d1_all)
            except ValueError:
                return (i-steps)
            
    return max_size
    
    
def CalculateMaxIterations(max_subsample_size, **kwargs):
    min_size = 10
    max_size = max_subsample_size
    flag=True
    step_size=5
    max_num_iters=15
    
    while flag:
        temp=(max_size-min_size)/step_size
        if temp.is_integer() and temp < max_num_iters:
            flag = False
        else:
            step_size=step_size+5
            
    num_iters = temp
    return step_size, num_iters


    
    