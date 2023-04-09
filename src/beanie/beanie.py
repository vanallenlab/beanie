import warnings

from tqdm.auto import tqdm

import os
import logging

import pandas as pd
import numpy as np
import scipy as sp
from scipy.sparse import issparse

import multiprocessing
from joblib import Parallel, delayed

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import upsetplot
from adjustText import adjust_text
from itertools import product
from statsmodels.stats.multitest import multipletests

import scanpy as sc

from . import driver_genes as dg
from . import differential_expression as de

from .utils import *

from .genesig import GeneSignature
from .scoring_aucell import aucell, create_rankings


# helper functions

def exists(val):
    return val is not None


# BEANIE class
class Beanie:
    
    def __init__(self, counts_path: str, metad_path: str, sig_path:str, normalised:bool, subsample_mode = "equal", matched_normals=False, bins=False, bin_size=None, min_cells=50, output_dir="./beanie_out/", **kwargs):
        """
        
        Parameters:
            counts_path                       path to counts matrix; must be (genes x cells) in .csv, .tsv or .h5ad format 
            metad_path                        path to the metadata; must contain 'group_id' and 'sample_id' columns; must be in .csv or .tsv format
            sig_path                          path to the file containing gene signatures. Must be .csv, .tsv or .gmt
            normalised                        boolean variable for indicating whether the matrix is normalised or not
            subsample_mode                    should be either "equal" or "max"
            matched_normals                   whether samples in the one of the groups are matched normals
            bins                              whether binning should be done for generating background gene signatures
            bin_size                          only applicable if bins=True. integer indicating size of bins
            min_cells                         minimum threshold for number of cells below which patients should be excluded from analysis
            output_dir                        directory for storing intermediate results
            
        Attributes:
            self.normalised_counts            (genes x cells) CPM normalised and log-transformed counts
            self.signatures                   file containing all the signature names and corresponding genes with header as signature names
            self.out                          output directory for intermediate results
            self.metad                        (cells x columns) metadata file containing columns for 'group_id', 'sample_id'
            self.signature_scores             (cells x signature_scores) matrix
            self.driver_genes                 dictionary of driver genes (default=10) for every signature
            self.subsampled_stats             dataframe storing the final results of subsampling procedure
            self.confidence_de_plot           figure for EstimateConfidenceDifferentialExpression
            self.barplot                      figure for PlotBarPlot function
            self.beeswarmplot                 figure for BeeSwarmPlot function
            self.heatmap                      figure for HeatmapDriverGenes function
            self.upsetplot_driver_genes       figure for UpsetPlotDriverGenes function
            self.upsetplot_signature_genes    figure for UpsetPlotSignatureGenes function
            self.de_obj                       DifferentialExpression object for max/custom subsample size
            self.de_summary                   dataframes containing the output of DifferentialExpression
            self.de_obj_simulation            list of DifferentialExpression objects for max/custom subsample siz
            self.de_summary_simulation        dictionary mapping the subsample size to dataframes generated from DifferentialExpression object 
            self.t1_ids                       list of patients belonging to treatment group A
            self.t2_ids                       list of patients belonging to treatment group B
            self.d1_all                       dictionary mapping patients to cell_ids in treatment group A
            self.d2_all                       dictionary mapping patients to cell_ids in treatment group B
            self.max_subsample_size
            self.group_id_names        list of treatment groups names in self.metad
            self.top_signatures               top 5 most significant and robust genes
            self.num_driver_genes             number of driver genes for which plots to be made
            self.t1_cells
            self.t2_cells
            self.patient_dropout_plot
            self.group_direction              direction of interest where enrichment should be seen
            self._candidate_robust_sigs_df
            self.subsample_mode               variable to decide whether smart-seq or 10x subsampling should be done
            self._null_dist_sigs              for storing dictionary of random gene sets corresponding to diff sizes of input sigs
            self._bins
            self._matched_normals          
            
        """
        ### check if all files exist/ have valid paths
        if not os.path.exists(counts_path):
            raise FileNotFoundError("Counts file does not exist. Please check input counts_path.")
        
        if not os.path.exists(metad_path):
            raise FileNotFoundError("Meta data file does not exist. Please check metad_path.")
        
        if not os.path.exists(sig_path):
            raise FileNotFoundError("Gene signatures file does not exist. Please check sig_path.")
         
        if subsample_mode not in list(["equal", "max"]):
            raise IOError("subsample_mode must be one of 'max' or 'equal'.")
        
        ### Read counts matrix
        logging.info("Reading counts matrix...")
        if counts_path.endswith(".csv"):
            counts = pd.read_csv(counts_path, index_col=0, sep=",")
            
        elif counts_path.endswith(".tsv"):
            counts = pd.read_csv(counts_path, index_col=0, sep="\t")
            
        elif counts_path.endswith(".h5ad"):
            sc_obj = sc.read(counts_path)
            if sc_obj.raw==None:
                if issparse(sc_obj.X):
                    counts = pd.DataFrame.sparse.from_spmatrix(sc_obj.X, index=sc_obj.obs_names, columns=sc_obj.var_names).T
                else:
                    counts = pd.DataFrame(sc_obj.X, index=sc_obj.obs_names, columns=sc_obj.var_names).T
            else:
                if issparse(sc_obj.raw.X):
                    counts = pd.DataFrame.sparse.from_spmatrix(sc_obj.raw.X, index=sc_obj.raw.obs_names, columns=sc_obj.raw.var_names).T
                else:
                    counts = pd.DataFrame(sc_obj.raw.X, index=sc_obj.raw.obs_names, columns=sc_obj.raw.var_names).T
        else:
            raise OSError("Counts matrix should be of the format .csv or .tsv, or a scanpy object .h5ad")

        # check that the index is not numbers
        if counts.index.dtype=="int64":
            raise OSError("Counts matrix format wrong. Please check.")

        
        ### Read meta data file
        logging.info("Reading meta data file...")
        if metad_path.endswith(".csv"):
            self.metad = pd.read_csv(metad_path, index_col=0, sep=",")
            try:
                self.metad = self.metad.loc[counts.columns,:]
            except:
                raise IOError("The cell ids in counts matrix and meta data file should match.")
        elif metad_path.endswith(".tsv"):
            self.metad = pd.read_csv(metad_path, index_col=0, sep="\t")
            try:
                self.metad = self.metad.loc[counts.columns,:]
            except:
                raise IOError("The cell ids in counts matrix and meta data file should match.")
        else:
            raise OSError("Meta data should be of the format .csv or .tsv")

        
        # Check if metadata file is in the correct format
        # check if the header has correct columns needed.
        if np.sum(self.metad.columns.isin(['group_id','sample_id']))!=2:
            raise IOError("Please check input format for meta data file. 'group_id','sample_id' columns should be present.")

        # check that number of treatment groups are 2
        self.group_id_names = sorted(set(self.metad.group_id))
                
        if len(self.group_id_names)<2:
            raise IOError("Atleast two treatment groups required.")
        elif len(self.group_id_names)>2:
            raise IOError("The method is not currently supported for more than two treatment groups.")
        
        self.output_dir = output_dir
        
        if exists(sig_path):
            logging.info("Reading signature file...")
            if sig_path.endswith(".csv"):
                self.signatures = pd.read_csv(sig_path, index_col=0, sep=",")
                self._writeSignatures()
            elif sig_path.endswith(".tsv"):
                self.signatures = pd.read_csv(sig_path, index_col=0, sep="\t")
                self._writeSignatures()
            elif sig_path.endswith(".gmt"):
                self._sig_path=sig_path
                with open(sig_path) as gmt:
                    ll = gmt.read()
                ll_sigs = ll.split("\n")
                if "" in ll_sigs:
                    ll_sigs.remove("")
                li = []
                for x in ll_sigs:
                    genes = x.split("\t")
                    li.append([i for i in genes if i])
                self.signatures = pd.DataFrame(li).T
                self.signatures.columns = self.signatures.iloc[0,:]
                self.signatures = self.signatures.iloc[1:,]
                self.signatures.index=range(len(self.signatures.index))    
            else:
                raise IOError("Signature file should be of the format .csv or .tsv or .gmt")

        else:
            raise IOError("File containing signature scores must be provided.")
            
        self.driver_genes = dict()
        self.num_driver_genes = 10
        self.subsample_mode = subsample_mode
        self._differential_expression_run = False
        self._driver_genes_run = False
        
        if bins == True:
            if exists(bin_size):
                if type(bin_size) is int:
                    self._bins = bin_size
                else:
                    raise IOError("bin_size must be an integer.")
            else:
                raise IOError("bin_size must be provided if bins=True.")
        else:
            self._bins = None
                
        self._matched_normals = matched_normals
        
        # check if matched normals, then both groups have same patients
        if self._matched_normals == True:
            p1 = set(self.metad.loc[self.metad.group_id == self.group_id_names[0],"sample_id"])
            p2 = set(self.metad.loc[self.metad.group_id == self.group_id_names[1],"sample_id"])
            if p1!=p2:
                raise IOError("Cells are not provided for each sample in both groups.")
                
        #check if there are any overlapping samples in the two treatment groups
        elif self._matched_normals == False:
            p1 = set(self.metad.loc[self.metad.group_id == self.group_id_names[0],"sample_id"])
            p2 = set(self.metad.loc[self.metad.group_id == self.group_id_names[1],"sample_id"])
            if len(p1.intersection(p2))!=0:
                raise IOError("Same sample_id present in both groups.")
            
        # remove patients with cells < 20
        cell_counts = self.metad.sample_id.value_counts()
        pats_below_threshhold = cell_counts[cell_counts<min_cells].index.to_list()
        if len(pats_below_threshhold)!=0:
            print("The following patients have less than "+str(min_cells)+" cells present, so they will be removed from analysis:", pats_below_threshhold)
            self.metad = self.metad[~self.metad.sample_id.isin(pats_below_threshhold)]
            counts = counts[self.metad.index]
            
        self.t1_ids = sorted(list(set(self.metad[self.metad.group_id==self.group_id_names[0]].sample_id)))
        self.t2_ids = sorted(list(set(self.metad[self.metad.group_id==self.group_id_names[1]].sample_id)))
        
        self.d1_all = {}
        for xid in self.t1_ids:
            self.d1_all[xid]=sorted(self.metad[self.metad.sample_id==xid]["sample_id"].index.to_list())
            
        self.d2_all = {}
        for xid in self.t2_ids:
            self.d2_all[xid]=sorted(self.metad[self.metad.sample_id==xid]["sample_id"].index.to_list())
            
        self.t1_cells = self.metad[self.metad.group_id == self.group_id_names[0]].index.to_list()
        self.t2_cells = self.metad[self.metad.group_id == self.group_id_names[1]].index.to_list()    
        
        if normalised==False:
            logging.info("Normalising counts...")
            self.normalised_counts = CPMNormalisationLogScaling(counts)
        else:
            self.normalised_counts = counts
            
            
        logging.info("Calculating maximum subsample size...")
        self.max_subsample_size = CalculateSubsampleSize(self.metad, self.group_id_names, self.subsample_mode, self._matched_normals)
#         self.max_subsample_size = self.metad.sample_id.value_counts()[-1]
        self.n_subsamples = min(int(self.metad.sample_id.value_counts()[0]/self.metad.sample_id.value_counts()[-1]),100)
        
        # REMOVE FOR FINAL UPLOAD
        self.signature_scores = None
        self._null_dist_scores = dict()
            
    def _writeSignatures(self):
        """Function to write signatures to a temporary file (.gmt) if they are of .csv/.tsv format. 
        Needed for beanie scoring.
        
        """
        
        dir_name = self.output_dir
        
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        
        # convert to GMT format (add an extra description column for the gene signatures)
        df = self.signatures.T
        df.insert(0,"description_col","NA")
        
        df.to_csv(dir_name + "/signatures.gmt", sep="\t", header=None)
        self._sig_path = dir_name + "/signatures.gmt"
        self._sig_path_dir = dir_name
        
        return

        
    def SignatureScoring(self, scoring_method="beanie", no_random_sigs=1000, aucell_quantile=0.05):
        """ 
        Function to do signature scoring using in-built scoring functions.
        
        Parameters:
            scoring_method                          choice between beanie (default), mean and combined-z to score the cells.
            no_random_sigs                          the number of random signatures that should be generated for FDR correction
            aucell_quantile                         parameter to indicate the quantile of genes to consider for ROC, if beanie method of scoring is being used. 
        
        """
        
        self._scoring_method = scoring_method
                       
        logging.info("Scoring signatures...")
             
        # Score background signatures
        sorted_genes = pd.Series.sort_values(self.normalised_counts.sum(axis=1))
        null_dist_sigs = GenerateNullDistributionSignatures(self.signatures, sorted_genes, self._bins, self.output_dir, no_random_sigs)
        self._null_dist_scores = dict()
        
        if scoring_method=="beanie":
            self._auc_cutoff = pd.Series(np.count_nonzero(self.normalised_counts.T, axis=1)).quantile(aucell_quantile)/self.normalised_counts.T.shape[1]
            signatures = GeneSignature.from_gmt(self._sig_path, field_separator='\t', gene_separator='\t')
            rnk_mtx = create_rankings(self.normalised_counts.T, seed=3120)
            self.signature_scores = aucell(rnk_mtx, signatures, auc_threshold=self._auc_cutoff, num_workers=multiprocessing.cpu_count())
            
            for key in tqdm(null_dist_sigs.keys()):
                signatures = GeneSignature.from_gmt(os.path.join(self.output_dir,"bg_signatures_" + str(key) + ".gmt"), field_separator='\t', gene_separator='\t')
                self._null_dist_scores[key] = aucell(rnk_mtx, signatures, auc_threshold=self._auc_cutoff, num_workers=multiprocessing.cpu_count())


#         elif scoring_method=="znorm":
#             self.signature_scores = SignatureScoringZNormalised(self.normalised_counts, self.signatures, multiprocessing.cpu_count())
#             for key in self._null_dist_sigs.keys():
#                 self._null_dist_scores[key] = SignatureScoringZNormalised(self.normalised_counts,
#                                                               pd.DataFrame(self._null_dist_sigs[key],
#                                                                            index=["random_sig_"+str(i) for i in range(len(self._null_dist_sigs[key]))]))
                                                                           
        elif scoring_method=="mean":
            self.signature_scores = SignatureScoringMean(self.normalised_counts,self.signatures)
            for key in self._null_dist_sigs.keys():
                self._null_dist_scores[key] = SignatureScoringMean(self.normalised_counts,
                                                              pd.DataFrame(self._null_dist_sigs[key],
                                                                           index=["random_sig_"+str(i) for i in range(len(self._null_dist_sigs[key]))]))

                     
        elif scoring_method=="combined-z":
            self.signature_scores = SignatureScoringCombinedZScore(self.normalised_counts,self.signatures)
            for key in self._null_dist_sigs.keys():
                self._null_dist_scores[key] = SignatureScoringCombinedZScore(self.normalised_counts,
                                                              pd.DataFrame(self._null_dist_sigs[key],
                                                                           index=["random_sig_"+str(i) for i in range(len(self._null_dist_sigs[key]))]))

            
        else:
            raise ValueError("Please choose scoring method from: 'beanie', 'mean', 'combined-z'.")

        return      
                
        
    def DifferentialExpression(self, cells_to_subsample_1=None, cells_to_subsample_2=None, alpha=0.05, min_ratio=0.9, subsamples=501, test_name="mwu-test", group_direction = None, **kwargs):
        """
        Function for finding out differentially expressed robust and statistically significant signatures. 
        
        Parameters: 
            cells_to_subsample                     cells that should be subsampled per patient; if no input provided, function to choose the max possible subsample size
            alpha                                  p-value cutoff
            min_ratio                              value of fold_rejection_ratio below which the signature is considered to be non-robust
            subsamples                             number of repeated subsamples in every fold
            minimum_expressing_samples             minimum number of samples that express gene to be considered
            minimum_frac_per_sample                minimum fraction of cells expressing for a gene to be considered expressed in a sample
            minimum_expression                     minimum expression value for a gene to be considered expressed in a cell
        
        """
        if self._differential_expression_run == True:
            print("DifferentialExpression has already been run.")
            return
        
        self._alpha = alpha
        self._min_ratio = min_ratio
        
        if test_name in ["mwu-test", "t-test", "ks-test", "kwh-test", "welch-test"]:
            self._de_test_name = test_name
        else:
            raise IOError("The 'test_name' must be one of 'mwu-test', 't-test', 'ks-test', 'kwh-test', 'welch-test'.") 
        
        correction_method = "fdr_bh"
        if cells_to_subsample_1 == None:
            cells_to_subsample_1 = self.max_subsample_size
        elif cells_to_subsample_1>self.max_subsample_size:
            warnings.warn("Too many cells to subsample in group1, switching to default max subsample size possible.", RuntimeWarning)
            cells_to_subsample_1 = self.max_subsample_size
            
        if cells_to_subsample_2 == None:
            cells_to_subsample_2 = self.max_subsample_size
        elif cells_to_subsample_2>self.max_subsample_size:
            warnings.warn("Too many cells to subsample in group2, switching to default max subsample size possible.", RuntimeWarning)
            cells_to_subsample_2 = self.max_subsample_size    
        

        if group_direction!=None:
            if group_direction not in self.group_id_names:
                raise IOError("The group_direction entered is not in the treatment groups. Please check group_direction.")
        else:
            group_direction = self.group_id_names[0]
        
        if exists(self._bins):
            sig_size_dict = {x:max(5,round(len(self.signatures[x].dropna())/self._bins)*self._bins) for x in self.signatures.columns}
        else:
            sig_size_dict = {x:"var" for x in self.signatures.columns}
        
        if self._matched_normals == False:
            self.de_obj = de.ExcludeSampleSubsampledDE(self.signature_scores.T, sig_size_dict, self._null_dist_scores,
                                                       self.d1_all, self.d2_all, self.subsample_mode,
                                                       group1_sample_cells=cells_to_subsample_1, 
                                                       group2_sample_cells=cells_to_subsample_2,
                                                       samples_per_fold=subsamples, test_name=test_name, **kwargs)

            self.de_obj.run()
            self.de_summary = self.de_obj.summarize(alpha, min_ratio)
            self.de_summary.insert(1,"corrected_p_inbuilt",multipletests(self.de_summary.p, method = correction_method)[1])

        elif self._matched_normals == True:
            self.de_obj = de.ExcludePatientPairedSubsampleDE(self.signature_scores.T, sig_size_dict, self._null_dist_scores,
                                                       self.d1_all, self.d2_all, self.subsample_mode,
                                                       group1_sample_cells=cells_to_subsample_1, 
                                                       group2_sample_cells=cells_to_subsample_2,
                                                       samples_per_fold=subsamples, test_name=test_name, **kwargs)

            self.de_obj.run()
            self.de_summary = self.de_obj.summarize(alpha, min_ratio)
            self.de_summary.insert(1,"corrected_p_inbuilt",multipletests(self.de_summary.p, method = correction_method)[1])
            
            
        # calculate other stats
        results_df = CalculateLog2FoldChangeSigScores(self.signature_scores, self.d1_all, self.d2_all)
        self.de_summary = pd.concat([results_df,self.de_summary], axis=1, sort=False)
        self.de_summary["direction"] = [self.group_id_names[0] if x==True else self.group_id_names[1] for x in self.de_summary.direction]

        # calculate the robust signatures in direction of interest: by default gr1 
        self._candidate_robust_sigs_df = self.de_summary[(self.de_summary.direction==group_direction) & (self.de_summary.nonrobust==False) & (self.de_summary.corr_p<=0.05)]
        
        robust_sigs = self._candidate_robust_sigs_df.sort_values(by=["log2fold","corr_p"],ascending=(False,True)).index
        
        if len(robust_sigs)>=5:
            self.top_signatures = robust_sigs[:5].to_list()
        else:
            self.top_signatures = robust_sigs.to_list()
            
        self._differential_expression_run = True
        
    def GetDifferentialExpressionSummary(self):
        if self._differential_expression_run==True:
            if self._sig_score_path==None:
                return self.de_summary[["log2fold","p","corr_p","corrected_p_inbuilt","nonrobust","direction"]]
            else:
                return self.de_summary[["log2fold","p","corr_p","nonrobust","direction"]]
        else:
            raise RuntimeError("Run DifferentialExpression() first.")
            
    def DriverGenes(self, group_direction=None):
        if self._driver_genes_run==True:
            print("DriverGenes() has already been run.")
            return
        
        logging.info("Finding Driver Genes...")
        
        if self._differential_expression_run==False:
            raise RuntimeError("Run DifferentialExpression() first.")

        if group_direction!=None:
            if group_direction not in self.group_id_names:
                raise IOError("The group_direction entered is not in the treatment groups. Please check group_direction.")
        else:
            group_direction = self.group_id_names[0]
                    
        for x in tqdm(self.signatures.columns):
            if group_direction==self.group_id_names[0]:
                if self.de_summary.loc[x,"direction"]==group_direction:
                    self.driver_genes[x] = dg.FindDriverGenes(x, self.signature_scores, self.normalised_counts.T, self.signatures[x].dropna().values, self.d1_all, self.d2_all)
                else:
                    self.driver_genes[x] = dg.FindDriverGenes(x, self.signature_scores, self.normalised_counts.T, self.signatures[x].dropna().values, self.d2_all, self.d1_all)
                    
            else:
                if self.de_summary.loc[x,"direction"]==group_direction:
                    self.driver_genes[x] = dg.FindDriverGenes(x, self.signature_scores, self.normalised_counts.T, self.signatures[x].dropna().values, self.d2_all, self.d1_all)
                else:
                    self.driver_genes[x] = dg.FindDriverGenes(x, self.signature_scores, self.normalised_counts.T, self.signatures[x].dropna().values, self.d1_all, self.d2_all)
                    
        self._driver_genes_run = True

    def GetDriverGenesSummary(self):
        
        if self._driver_genes_run==False:
            raise RuntimeError("Run DriverGenes() method first.")
            
        elif self._differential_expression_run==False:
            raise RuntimeError("Run DifferentialExpression() first.")
        
        tup = []
        for k in self.driver_genes.keys():
            try:
                v = self.driver_genes[k].index
                for gene in v:
                    tup.append((k,self.de_summary.loc[k,"direction"],gene))
            except AttributeError:
                pass
            
        df = pd.DataFrame(pd.concat(self.driver_genes.values()).values, index = pd.MultiIndex.from_tuples(tup), 
             columns = ["gr1_outlier","gr2_outlier","log2fold","log2fold_outlier","std_error","direction","robustness_ratio"])
        
        return df[["log2fold","std_error","robustness_ratio","log2fold_outlier","gr1_outlier","gr2_outlier"]]


    def BarPlot(self, hatch_color = '#FFFFFF', dpi_res = 300, color_gr1 = "#1E8449", color_gr2 = "#7D3C98", alpha_val = 0.5, **kwargs):
        """
        Function for generating barplot for statistically significant pathways (robust and non-robust).
        Hatched bars represent signatures with statistically significant differences between groups but non-robust to subsampling.
        Solid bars represent robust signatures with statistically significant differences.

        """

        # define all kwargs parameters

        if self._differential_expression_run==False:
            raise RuntimeError("Run DifferentialExpression() first.")

        flag=0
        df_plot = self.de_summary[["p","statistic","corr_p","nonrobust","log2fold","direction"]]
        if df_plot.loc[df_plot.corr_p<=0.05,:].shape[0]!=0:
            df_plot = df_plot.loc[df_plot.corr_p<=0.05,:]
        else:
            flag=1
            print("No significant signature found...")
            return

        # find number of significant digits
        keys = sorted(self.de_obj.null_dist_folds.keys())
        self._significant_digits = len(str(len(self.de_obj.null_dist_folds[keys[0]][0].p.stack().values)))-1

        # make values non-zero for taking log
        a = df_plot.corr_p.copy()
        for i in range(len(a)):
            if a[i]==0:
                a[i] += 1/pow(10,self._significant_digits)
        df_plot["corr_p_modified"] = (a).astype(float)

        df_plot["log_corrp"] = -np.log10(df_plot["corr_p_modified"])
        df_plot["log_corrp"] = [df_plot["log_corrp"][count] if df_plot["direction"][count] == self.group_id_names[0] else -1*df_plot["log_corrp"][count] for count in range(0,df_plot.shape[0])]
        df_plot = df_plot.sort_values(by=["log_corrp"],axis=0)
        size = df_plot.shape[0]

        plt.rcParams['hatch.color'] = hatch_color

        with sns.plotting_context("notebook", rc={'axes.titlesize' : 10,
                                               'axes.labelsize' : 10,
                                               'xtick.labelsize' : 10,
                                               'ytick.labelsize' : 12,
                                               'font.name' : u'Arial'}):
            fig, axs = plt.subplots(dpi=dpi_res)
            bar = sns.barplot(x=df_plot.index ,y= "log_corrp", data=df_plot, color=color_gr1, **kwargs)
            for i,thisbar in enumerate(bar.patches):

                #set different color for bars which are up in treatment-group2
                if df_plot["log_corrp"][i]<0:
                    thisbar.set_color(color_gr2)

                # Set a different hatch for bars which are non-robust
                if df_plot["nonrobust"][i]:
                    thisbar.set_hatch("\\")
                    thisbar.set_alpha(alpha_val)

            if flag==1:
                plt.hlines(linestyles='dashed',y=np.log10(0.05), xmin=-0.5, xmax=df_plot.shape[0]-0.5,colors=".3")
                plt.hlines(linestyles='dashed',y=-np.log10(0.05), xmin=-0.5, xmax=df_plot.shape[0]-0.5,colors=".3")

            if flag==0:    
                axs.set_title("Statistically significant signatures")
            axs.set_ylabel("Empirical log p-value")
            axs.set_xlim(left=-0.5,right=df_plot.shape[0]-0.5)
            axs.set_ylim(bottom = -(self._significant_digits+1), top = self._significant_digits+1)

            circ1 = mpatches.Patch(facecolor="#B2B1B0", alpha=alpha_val, hatch='\\\\', label='Non-robust to subsampling')
            circ2 = mpatches.Patch(facecolor="#B2B1B0", alpha=alpha_val, label='Robust to subsampling')
            circ3 = mpatches.Patch(facecolor=color_gr1, label='Enriched in '+self.group_id_names[0])
            circ4 = mpatches.Patch(facecolor=color_gr2, label='Enriched in '+self.group_id_names[1])
            axs.legend(handles = [circ1,circ2,circ3,circ4], bbox_to_anchor=(1.01, 1), loc='upper left')

            # Add *** above bars which have p-val<1/significant_digits
            count = 0
            for p in axs.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                if abs(_y)==self._significant_digits and df_plot.iloc[count,:].corr_p==0:
                    if _y<0:
                        axs.text(_x, _y-0.5, "**", ha="center")
                    else:
                        axs.text(_x, _y+0.5, "**", ha="center")
            plt.xticks(rotation=90);

            self.barplot = fig
            fig.savefig(os.path.join(self.output_dir,"barplot.png"))
        return
    
    def PatientDropoutPlot1(self, annotate=True):
        """
        Function to Matrix for whether a patient 

        """

        if self._differential_expression_run==False:
            raise RuntimeError("Run DifferentialExpression() first.")

        if self.de_summary[(self.de_summary.nonrobust==True) & (self.de_summary.corr_p<=0.05)].shape[0]==0:
            print("No significant signatures found...")
            non_robust_sig = self.de_summary[(self.de_summary.nonrobust==True)]
        else:
            non_robust_sig = self.de_summary[(self.de_summary.nonrobust==True) & (self.de_summary.corr_p<=0.05)]
        non_robust_sig_plot = non_robust_sig[self.de_summary.columns[self.de_summary.columns.str.startswith("excluded")]]
        non_robust_sig_plot.columns = [x[9:] for x in non_robust_sig_plot.columns]

        size=non_robust_sig_plot.shape[0]

        # Label the outliers
        q25 = non_robust_sig_plot.T.quantile(0.25).to_numpy()
        q75 = non_robust_sig_plot.T.quantile(0.75).to_numpy()
        outlier_bottom_lim = q25 - 1.5 * (q75 - q25)

        mat = pd.DataFrame(index = non_robust_sig_plot.index, columns = non_robust_sig_plot.columns)
        for i in range(non_robust_sig_plot.shape[0]):
            fr_ratio = non_robust_sig_plot.iloc[i,:]
            pats = [x for x in fr_ratio[fr_ratio<self._min_ratio].index.to_list()]
            mat.iloc[i,:] = mat.columns.isin(pats).astype(int)

        # sort matrix accoring to frequency (rowsum, colsum)
        mat.loc["sum",:] = mat.sum(axis=0)
        mat = mat.T.sort_values(by=["sum"]).T.drop(["sum"])

        mat["sum"] = mat.sum(axis=1)
        mat = mat.sort_values(by=["sum"]).drop(["sum"], axis=1)
        mat = mat.astype(int)

        df_main = non_robust_sig_plot.loc[mat.index,mat.columns]

        with sns.plotting_context("notebook", rc={'axes.titlesize' : 12,
                                           'axes.labelsize' : 10,
                                           'xtick.labelsize' : 14,
                                           'ytick.labelsize' : 14,
                                           'font.name' : u'Arial'}):
            fig = plt.figure(figsize=(1.25*df_main.shape[0],1.25*df_main.shape[1]/2.5),dpi=300)
            gs = GridSpec(4,4)
            gs.update(wspace=0.015, hspace=0.05)
            ax_main = plt.subplot(gs[1:4, :3])
            ax_yDist = plt.subplot(gs[1:4, 3], sharey=ax_main)
            ax_xDist = plt.subplot(gs[0:1, :3])

            sns.boxplot(data=df_main.T, color="#6CC9B6",ax=ax_xDist)
            ax_xDist.xaxis.set_tick_params(which='both', labeltop=False, labelbottom=False)
            ax_xDist.set_ylabel("FRR",size=14);
            ax_xDist.set_ylim(bottom=-0.5)

            heatmap = sns.heatmap(mat.T, cmap="RdPu",ax=ax_main, cbar=False, vmin=0,vmax=2.0, linewidths=0.5, linecolor="#F7DD8B")
            ax_main.set_yticks([])
            ax_main.set_yticks(np.arange(mat.T.shape[0])+0.5)
            ax_main.set_yticklabels(mat.T.index.to_list());
            ax_main.set_xticks(np.arange(mat.T.shape[1])+0.5)
            ax_main.set_xticklabels(mat.T.columns.to_list(), rotation=90);
            for _, spine in heatmap.spines.items():
                spine.set_visible(True)

            ax_yDist.barh(np.arange(mat.shape[1])+0.5,mat.T.sum(axis=1),color="#FED298")
            ax_yDist.yaxis.set_tick_params(which='both', labelright=False, labelleft=False)
            ax_yDist.set_xlim([0,mat.T.shape[1]+1]);
            ax_yDist_twin = ax_yDist.twinx()
            ax_yDist_twin.set_ylabel("No. of dropout signatures per patient", size=16, loc="center")

            if annotate==True:
                q25 = df_main.T.quantile(0.25).to_numpy()
                q75 = df_main.T.quantile(0.75).to_numpy()
                outlier_top_lim = q75 + 1.5 * (q75 - q25)
                outlier_bottom_lim = q25 - 1.5 * (q75 - q25)
                texts = []
                for i in range(df_main.shape[0]):
                    fr_ratio = df_main.iloc[i,:]
                    pats = [x for x in fr_ratio[fr_ratio<outlier_bottom_lim[i]].index.to_list()]
                    texts.extend([ax_xDist.text(i,fr_ratio[fr_ratio<outlier_bottom_lim[i]][j],pats[j],ha='center', va='center',fontsize=9) for j in range(len(pats))])

                adjust_text(texts,arrowprops=dict(arrowstyle='-',color='#014439'));
            circ1 = mpatches.Patch(facecolor="#F34C92", alpha=0.9, linewidth=1, label='FRR below robustness threshhold')
            circ2 = mpatches.Patch(facecolor="#F6E0A3", alpha=0.4, linewidth=1, label='FRR above robustness threshhold')
            fig.legend(handles = [circ1,circ2], bbox_to_anchor=(1,0), loc='upper right')


        self.patient_dropout_plot = fig
        fig.savefig(os.path.join(self.output_dir,"patient_exclusion_plot.png"))
        return

    def HeatmapDriverGenes(self, signature_names=None, num_genes = 10, **kwargs):
        """
        Function for plotting driver genes (by default top 10) of all robust signatures.
        thought - is it possible to incorporate corr-coeff, pval as well in this plot
        
        Parameters:
            signature_names                      names of signatures for which heatmap has to be plotted
        
        """
        
        if signature_names == None:
            if self._differential_expression_run==False:
                raise RuntimeError("Run DifferentialExpression() first.")
            else:
                signature_names = self.top_signatures
                
        if self._driver_genes_run==False:
            raise RuntimeError("Run DriverGenes() first.")
            
        self.num_driver_genes = num_genes
        self.heatmap = dg.GenerateHeatmap(self.normalised_counts.T, self.t1_ids, self.t2_ids, self.d1_all, self.d2_all, self.driver_genes, signature_names, num_genes, **kwargs)
        fig.savefig(os.path.join(self.output_dir,"heatmap.png"))
        return
        
    def UpsetPlotDriverGenes(self, fig_width=None, signature_names=None):
        """
        Function to plot intersection of driver genes between different signatures 
        (option to either enter the signature names list for which you want the plot; 
        by default will pick the top5 signatures).
        
        Parameters:
            signature_names                     (optional) names of signatures for which upsetplot should be plotted 
        
        """
                        
        if signature_names == None:
            if self._differential_expression_run==False:
                raise RuntimeError("Run DifferentialExpression() first.")
            else:
                signature_names = self.top_signatures

        elif len(signature_names)>6:
            print("Too many signature names to show upset plot")
            
        if self._driver_genes_run==False:
            raise RuntimeEror("Run DriverGenes() first.")
            
        upset_df_prep = pd.DataFrame(columns=self.driver_genes.keys())
        for x in self.driver_genes.keys():
            try:
                a = self.driver_genes[x].index.to_list()
                a = a[:min(len(a),self.num_driver_genes)]
                a += [None] * (self.num_driver_genes - len(a))
                upset_df_prep[x] = a
            except AttributeError:
                pass

        upset_df = pd.DataFrame(list(product([True,False],repeat = len(signature_names))),columns=signature_names)
        intersection = list()
        for i in range(upset_df.shape[0]):
            sig_names = upset_df.columns[upset_df.iloc[i,]]
            temp = upset_df_prep[sig_names]
            list_sigs = [set(temp.iloc[:,i].dropna()) for i in range(temp.shape[1])]
            if len(list_sigs)!=0:
                intersection.append(len(set.intersection(*list_sigs)))
            else:
                intersection.append(0)
        upset_df["Intersection"] = intersection
        upset_df = upset_df.groupby(by = signature_names).sum()
        
        if fig_width==None:
            if len(signature_names)==6:
                fig_width = 25
            elif len(signature_names)==5:
                fig_width = 15
            elif len(signature_names)==4:
                fig_width = 10
            elif len(signature_names)==3:
                fig_width = 7
        fig_height = 5
        
        fig, axs = plt.subplots(1,1, figsize=(fig_width,fig_height), dpi=300)
        self.upsetplot_driver_genes = upsetplot.plot(upset_df['Intersection'], sort_by='cardinality', fig=fig, element_size=None, show_counts=True)
        axs.axis('off');
        fig.suptitle("Overlap between top ranked genes")
        fig.savefig(os.path.join(self.output_dir,"upsetplot_topkgenes.png"))
        return
        
    def UpsetPlotSignatureGenes(self, fig_width = None, signature_names=None):
        """
        Function to plot intersection of genes in every signature.
        
        Parameters:
            top_features            list of signatures for which to plot upsetplot
        
        """
                         
        if signature_names == None:
            if self.top_signatures==None:
                print("Please run DifferentialExpression() method first")
                return
            else:
                signature_names = self.top_signatures
        elif len(signature_names)>6:
            print("Too many signature names to show upset plot")

        upset_df = pd.DataFrame(list(product([True,False],repeat = len(signature_names))),columns=signature_names)
        intersection = list()
        for i in range(upset_df.shape[0]):
            sig_names = upset_df.columns[upset_df.iloc[i,]]
            temp = self.signatures[sig_names]
            list_sigs = [set(temp.iloc[:,i].dropna()) for i in range(temp.shape[1])]
            if len(list_sigs)!=0:
                intersection.append(len(set.intersection(*list_sigs)))
            else:
                intersection.append(0)
        upset_df["Intersection"] = intersection
        upset_df = upset_df.groupby(by = signature_names).sum()
        
        if fig_width==None:
            if len(signature_names)==6:
                fig_width = 25
            elif len(signature_names)==5:
                fig_width = 15
            elif len(signature_names)==4:
                fig_width = 10
            elif len(signature_names)==3:
                fig_width = 7
        fig_height = 5
        
        fig, axs = plt.subplots(1,1,figsize=(fig_width,fig_height), dpi=300)
        self.upsetplot_signature_genes = upsetplot.plot(upset_df['Intersection'], sort_by='cardinality', fig=fig, element_size=None, show_counts=True)
        axs.axis('off');
        fig.suptitle("Overlap between gene signatures")
        
        fig.savefig(os.path.join(self.output_dir,"upsetplot_signatures.png"))
        return
    
#     def EstimateConfidenceDifferentialExpression(self, alpha=0.05, min_ratio=0.9,
#                                                  subsamples=501, **kwargs):
#         """
#         Function for generating saturation curve for simulation. Helps in estimating the confidence levels 
        
#         Parameters: 
#             alpha                                  p-value cutoff
#             min_ratio                              value of fold_rejection_ratio below which the signature is considered to be non-robust
#             subsamples                             number of repeated subsamples in every fold
#             minimum_expressing_samples             minimum number of samples that express gene to be considered
#             minimum_frac_per_sample                minimum fraction of cells expressing for a gene to be considered expressed in a sample
#             minimum_expression                     minimum expression value for a gene to be considered expressed in a cell
        
#         """
#         if len(self.de_obj_simulation)!=0:
#             print("Simulation has already been run.")
#             return
        
#         step_size,max_iters = CalculateMaxIterations(self.max_subsample_size)
        
#         for i in tqdm(range(0,max_iter)):
#             self.de_obj_simulation.append(de.ExcludeSampleSubsampledDENoCorrection(self.signature_scores.T, 
#                                                            self.d1_all, self.d2_all, self.subsample_mode, 
#                                                            group1_sample_cells=10+step_size*i, 
#                                                            group2_sample_cells=10+step_size*i, 
#                                                            samples_per_fold=subsamples, **kwargs))
#             self.de_obj_simulation[i].run()
#             self.de_summary_simulation.append(self.de_obj_simulation[i].summarize(alpha, min_ratio))
            
            
#     def PlotConfidenceDifferentialExpression(self):
#         """
#         Function to plot number of signatures which are robust, non-robust, statistically significant, for different
#         number of cells subsampled per patient.
        
#         """
        
#         robust_and_sig = []
#         nonrobust_and_sig = []
#         for i in range(len(self.de_summary_simulation)):
#             robust_and_sig.append(self.de_summary_simulation[i][(self.de_summary_simulation[i].nonrobust==False) & (self.de_summary_simulation[i].p<=0.05)].shape[0])
#             nonrobust_and_sig.append(self.de_summary_simulation[i][(self.de_summary_simulation[i].nonrobust==True) & (self.de_summary_simulation[i].p<=0.05)].shape[0])
    
#         conf_list = []
#         for i in tqdm(range(len(self.de_obj_simulation))):
#             temp_list = []
#             temp_obj = self.de_obj_simulation[i]
#             temp_list.extend([(temp_obj.folds[j].p<0.05).sum() for j in range(len(temp_obj.folds))])
#             conf_list.append(temp_list)
            
#         flat_list = []
#         for i in range(len(conf_list)):
#             flat_list.append([item for sublist in conf_list[i] for item in sublist])
            
#         boxplot_df = pd.DataFrame(flat_list)
        
#         fig, axs = plt.subplots(figsize=(len(conf_list)/2,5))

#         step_size,max_iters = CalculateMaxIterations(self.max_subsample_size)

#         bp = axs.boxplot(boxplot_df,labels=[10+step_size*i for i in range(0,len(conf_list))],patch_artist=True);

#         for box in bp['boxes']:
#             box.set(color='#FB91A4', linewidth=2)
#             box.set(facecolor = '#F5D9DD')
#         for whisker in bp['whiskers']:
#             whisker.set(color='#7570B3', linewidth=2)
#         for cap in bp['caps']:
#             cap.set(color='#7570B3', linewidth=2)
#         for median in bp['medians']:
#             median.set(color='#F3497F', linewidth=2)
#         for flier in bp['fliers']:
#             flier.set(marker='.', color='#E7298A', alpha=0.5)

#         axs.plot(range(1,len(conf_list)+1),robust_and_sig, label="robust", color="#F3497F", marker="o")
#         axs.plot(range(1,len(conf_list)+1),nonrobust_and_sig, label="non-robust", color="#C3A5E0", marker=".", linestyle='dashed', alpha=0.7)
#         axs.plot([1,len(conf_list)+1],
#                  [self.de_summary_simulation[0].shape[0],
#                   self.de_summary_simulation[0].shape[0]],color="#000000", linestyle='dashed', label="total")
#         axs.legend(bbox_to_anchor=(1.01, 1), loc='upper left');
#         axs.set_xlabel("Number of cells subsampled per patient")
#         axs.set_ylabel("Number of Signatures")
#         axs.set_title("Confidence Level Estimation");
#         self.confidence_de_plot = fig
#         fig.savefig(os.path.join(self.output_dir,"confidence_plot_de.png"))
#         return    