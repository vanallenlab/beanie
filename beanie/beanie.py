from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import multiprocessing

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


class Beanie:
    
    def __init__(self, counts_path: str, metad_path: str, normalised: bool, test_direction: str, sig_path = None, sig_score_path = None, subsample_mode = "equal", **kwargs):
        """
        
        Parameters:
            counts_path                       path to the normalised counts_matrix file
            metad_path                        path to the metadata file
            normalised
            test_direction
            sig_path                          path to the signature file containing
            sig_db                            arguments for extracting information from msigdb. In this case, sig_path will not be needed
            sig_score_path                    path to signature_scores


            Attributes:
            self.counts                       (genes x cells) counts matrix in .csv format
            self.normalised_counts            CPM and log-scaling of counts
            self.signatures                   file containing all the signature names and corresponding genes with header as signature names
            self.metad                        (cells x columns) metadata file containing columns for 'treatment_group', 'patient_id'
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
            self.treatment_group_names        list of treatment groups names in self.metad
            self.top_signatures               top 5 most significant and robust genes
            self.num_driver_genes             number of driver genes for which plots to be made
            self.t1_cells
            self.t2_cells
            self.patient_dropout_plot
            self.group_direction              direction of interest where enrichment should be seen
            self.__candidate_robust_sigs_df
            self.subsample_mode               variable to decide whether smart-seq or 10x subsampling should be done
            
        """
        
        # Read counts matrix
        try:
            print("************************************************************")
            print("Reading counts matrix...")
            if counts_path.endswith(".csv"):
                self.counts = pd.read_csv(counts_path, index_col=0, sep=",")
            elif counts_path.endswith(".tsv"):
                self.counts = pd.read_csv(counts_path, index_col=0, sep="\t")
            elif counts_path.endswith(".h5ad"):
                sc_obj = sc.read(counts_path)
                if sc_obj.raw==None:
                    self.counts = pd.DataFrame(sc_obj.X, index=sc_obj.obs_names, columns=sc_obj.var_names).T
                else:
                    self.counts = pd.DataFrame(sc_obj.raw.X, index=sc_obj.raw.obs_names, columns=sc_obj.raw.var_names).T
            else:
                print("Counts matrix should be of the format .csv or .tsv")
                return
            
            # check that the index is not numbers
            if self.counts.index.dtype=="int64":
                print("Gene names cannot be integers. Please check input format for counts matrix.")
                return
        
        except FileNotFoundError:
            print("Counts file does not exist. Please check input counts_path.")
            print("************************************************************")
            return
        
        # Read meta data file
        try:
            print("Reading meta data file...")
            if metad_path.endswith(".csv"):
                self.metad = pd.read_csv(metad_path, index_col=0, sep=",")
            elif metad_path.endswith(".tsv"):
                self.metad = pd.read_csv(metad_path, index_col=0, sep="\t")
            else:
                print("Meta data should be of the format .csv or .tsv")
                return
        
        except FileNotFoundError:
            print("Meta data file does not exist. Please check metad_path.")
            print("************************************************************")
            return
        
        # Check if metadata file is in the correct format
        
        # check if the header has correct columns needed.
        if np.sum(self.metad.columns.isin(['treatment_group','patient_id']))!=2:
            print("Please check input format for meta data file. 'treatment_group','patient_id' columns should be present.")
            return

        # check that number of treatment groups are 2
        self.treatment_group_names = sorted(set(self.metad.treatment_group))
                
        if len(self.treatment_group_names)<2:
            print("Atleast two treatment groups required.")
            return
        elif len(self.treatment_group_names)>2:
            print("The method is not currently supported for more than two treatment groups.")
            return
        else:
            if test_direction not in self.treatment_group_names:
                print("Check input parameter test_direction")
                return 
            else:
                self.group_direction = test_direction
                if test_direction!=self.treatment_group_names[0]:
                    self.treatment_group_names.reverse()
                
        # check if cell ids in self.normalised_counts and self.metad match
        if set(self.counts.columns) != set(self.metad.index):
            print("The cell ids in counts matrix and meta data file should match.")
            return
        
        
        try:
            if sig_path!= None:
                print("Reading signature file...")
                if sig_path.endswith(".csv"):
                    self.signatures = pd.read_csv(sig_path, sep=",")
                elif sig_path.endswith(".tsv"):
                    self.signatures = pd.read_csv(sig_path, sep="\t")
                else:
                    print("Signature file should be of the format .csv or .tsv")
                    return

                # check that first column is not integers
                if self.signatures.iloc[:,0].dtype =="int64":
                    self.signatures = pd.read_csv(sig_path, index_col=0)
            
            # If signatures have to be extracted from MsigDb
            else:
                self.signatures = GetSignaturesMsigDb(**kwargs)
                
        except FileNotFoundError:
            print("Signature file does not exist. Please check sig_path.")
            print("************************************************************")
            return

        # Read signature scores file
        try:
            if sig_score_path!=None:
                print("Reading signature scores...")
                if sig_score_path.endswith(".csv"):
                    self.signature_scores = pd.read_csv(sig_score_path, index_col=0, sep=",")
                elif sig_score_path.endswith(".tsv"):
                    self.signature_scores = pd.read_csv(sig_score_path, index_col=0, sep="\t")
                else:
                    print("Signature score file should be of the format .csv or .tsv")
                    return
                
                # check if self.signature_scores has the same cell ids as self.normalised_counts
                if set(self.signature_scores.index) != set(self.counts.columns):
                    print("The cell ids in counts matrix and signature scores matrix should match.")
                
            else:
                self.signature_scores = None
        
        except FileNotFoundError:
            print("Signature scores file does not exist. Please check sig_score_path.")
            print("************************************************************")
            return
        
            
        self.driver_genes = dict()
        self.subsampled_stats = None
        self.confidence_de_plot = None
        self.barplot = None
        self.beeswarmplot = None
        self.heatmap = None
        self.upsetplot_driver_genes = None
        self.upsetplot_signature_genes = None
        self.de_obj = None
        self.de_summary = None
        self.de_obj_simulation = list()
        self.de_summary_simulation = list()
        self.top_signatures=None
        self.num_driver_genes=10
        self.patient_dropout_plot=None  
        self.subsample_mode = subsample_mode
        
        self.t1_ids = list(set(self.metad[self.metad.treatment_group==self.treatment_group_names[0]].patient_id))
        self.t2_ids = list(set(self.metad[self.metad.treatment_group==self.treatment_group_names[1]].patient_id))
        
        self.d1_all = {}
        for xid in self.t1_ids:
            self.d1_all[xid]=self.metad[self.metad.patient_id==xid]["patient_id"].index.to_list()
            
        self.d2_all = {}
        for xid in self.t2_ids:
            self.d2_all[xid]=self.metad[self.metad.patient_id==xid]["patient_id"].index.to_list()
            
        self.t1_cells = self.metad[self.metad.treatment_group == self.treatment_group_names[0]].index.to_list()
        self.t2_cells = self.metad[self.metad.treatment_group == self.treatment_group_names[1]].index.to_list()    
        
        if normalised==False:
            self.normalised_counts = CPMNormalisationLogScaling(self.counts)
        else:
            self.normalised_counts = self.counts
        self.max_subsample_size = CalculateSubsampleSize(self.metad, self.treatment_group_names, self.subsample_mode)
        
        print("************************************************************")
        
        
    def SignatureScoring(self, scoring_method="mean"):
        """ 
        Function to do signature scoring using in-built scoring functions.
        
        Parameters:
            scoring_method                          choice between mean and znorm to score the cells.
        
        """
        
        
        ## TODO: add check if signature_scores already exists
        
        if scoring_method=="znorm":
            self.signature_scores = SignatureScoringZNormalised(self.normalised_counts, self.signatures, multiprocessing.cpu_count())
        elif scoring_method=="mean":
            self.signature_scores = SignatureScoringMean(self.normalised_counts,self.signatures)
        elif scoring_method=="combined-z":
            self.signature_scores = SignatureScoringCombinedZScore(self.normalised_counts,self.signatures)
        else:
            print("Please choose scoring method from: 'znorm', 'mean', 'combined-z'.")
            return

        
    def DifferentialExpression(self, cells_to_subsample_1=None, cells_to_subsample_2=None, alpha=0.05, min_ratio=0.9, subsamples=501, **kwargs):
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
        
        correction_method = "bonferroni"
        if cells_to_subsample_1 == None:
            cells_to_subsample_1 = self.max_subsample_size
            
        if cells_to_subsample_2 == None:
            cells_to_subsample_2 = self.max_subsample_size
        
        self.de_obj = de.ExcludeSampleSubsampledDE(self.signature_scores.T, 
                                               self.d1_all, self.d2_all, self.subsample_mode,
                                               group1_sample_cells=cells_to_subsample_1, 
                                               group2_sample_cells=cells_to_subsample_2,
                                               samples_per_fold=subsamples,**kwargs)
        self.de_obj.run()
        self.de_summary = self.de_obj.summarize(alpha, min_ratio)
        # correct for multiple hypothesis
        self.de_summary.insert(1,"corrected_p",multipletests(self.de_summary.p, method = correction_method)[1])

        # calculate other stats
        results_df = CalculateLog2FoldChangeSigScores(self.signature_scores, self.d1_all, self.d2_all)
        self.de_summary = pd.concat([results_df,self.de_summary],axis=1, sort=False)
        self.de_summary["direction"] = [self.treatment_group_names[0] if x==True else self.treatment_group_names[1] for x in self.de_summary.direction]

        # calculate the robust signatures in direction of interest: by default gr1 
        self.__candidate_robust_sigs_df = self.de_summary[(self.de_summary.direction==self.group_direction) & (self.de_summary.nonrobust==False) & (self.de_summary.corrected_p<=0.05)]
        
        robust_sigs = self.__candidate_robust_sigs_df.sort_values(by=["log2fold","corrected_p"],ascending=(False,True)).index
        
        if len(robust_sigs)>=5:
            self.top_signatures = robust_sigs[:5].to_list()
        else:
            self.top_signatures = robust_sigs.to_list()
            
    def GetDifferentialExpressionSummary():
        return self.de_summary
            
    def DriverGenes(self, num_genes=10):
        print("************************************************************")
        print("Finding Driver Genes")
        
        ## TODO: ADD CONDITION TO CHECK IF DIFFERENTIAL EXPRESSION HAS ALREADY BEEN RUN
#         if not self.de_summary:
#             print("Run DifferentialExpression() first")
#             print("************************************************************")
#             return
        
        self.num_driver_genes=num_genes
        
        for x in tqdm(self.signatures.columns):
            if self.de_summary.loc[x,"direction"]==self.group_direction:
                self.driver_genes[x] = dg.FindDriverGenes(x, self.signature_scores, self.normalised_counts.T, self.signatures[x].dropna().values, self.d1_all, self.d2_all, num_genes)
            else:
                self.driver_genes[x] = dg.FindDriverGenes(x, self.signature_scores, self.normalised_counts.T, self.signatures[x].dropna().values, self.d2_all, self.d1_all, num_genes)
        print("************************************************************")

    def GetDriverGenesSummary(self):
        
        if len(self.driver_genes)==0:
            print("Run DriverGenes() method first.")
            pass
        elif self.de_summary==None:
            print("Run DifferentialExpression() method first.")
            pass
        
        tup = []
        for k in self.driver_genes.keys():
            try:
                v = self.driver_genes[k].index
                for gene in v:
                    tup.append((k,self.de_summary.loc[k,"direction"],gene))
            except AttributeError:
                pass
            
        return pd.DataFrame(pd.concat(self.driver_genes.values()).values, index = pd.MultiIndex.from_tuples(tup), 
             columns = ["log2fold","std_error","direction","robustness_ratio"])
        
    
    def EstimateConfidenceDifferentialExpression(self, alpha=0.05, min_ratio=0.9,
                                                 subsamples=501, **kwargs):
        """
        Function for generating saturation curve for simulation. Helps in estimating the confidence levels 
        
        Parameters: 
            alpha                                  p-value cutoff
            min_ratio                              value of fold_rejection_ratio below which the signature is considered to be non-robust
            subsamples                             number of repeated subsamples in every fold
            minimum_expressing_samples             minimum number of samples that express gene to be considered
            minimum_frac_per_sample                minimum fraction of cells expressing for a gene to be considered expressed in a sample
            minimum_expression                     minimum expression value for a gene to be considered expressed in a cell
        
        """
        if len(self.de_obj_simulation)!=0:
            print("Simulation has already been run.")
            return
        
        step_size,max_iters = CalculateMaxIterations(self.max_subsample_size)
        
        for i in tqdm(range(0,max_iter)):
            self.de_obj_simulation.append(de.ExcludeSampleSubsampledDE(self.signature_scores.T, 
                                                           self.d1_all, self.d2_all, self.subsample_mode, 
                                                           group1_sample_cells=10+step_size*i, 
                                                           group2_sample_cells=10+step_size*i, 
                                                           samples_per_fold=subsamples, **kwargs))
            self.de_obj_simulation[i].run()
            self.de_summary_simulation.append(self.de_obj_simulation[i].summarize(alpha, min_ratio))
            
            
    def PlotConfidenceDifferentialExpression(self):
        """
        Function to plot number of signatures which are robust, non-robust, statistically significant, for different
        number of cells subsampled per patient.
        
        """
        
        robust_and_sig = []
        nonrobust_and_sig = []
        for i in range(len(self.de_summary_simulation)):
            robust_and_sig.append(self.de_summary_simulation[i][(self.de_summary_simulation[i].nonrobust==False) & (self.de_summary_simulation[i].p<=0.05)].shape[0])
            nonrobust_and_sig.append(self.de_summary_simulation[i][(self.de_summary_simulation[i].nonrobust==True) & (self.de_summary_simulation[i].p<=0.05)].shape[0])
    
        conf_list = []
        for i in tqdm(range(len(self.de_obj_simulation))):
            temp_list = []
            temp_obj = self.de_obj_simulation[i]
            temp_list.extend([(temp_obj.folds[j].p<0.05).sum() for j in range(len(temp_obj.folds))])
            conf_list.append(temp_list)
            
        flat_list = []
        for i in range(len(conf_list)):
            flat_list.append([item for sublist in conf_list[i] for item in sublist])
            
        boxplot_df = pd.DataFrame(flat_list)
        
        fig, axs = plt.subplots(figsize=(len(conf_list)/2,5))

        step_size,max_iters = CalculateMaxIterations(self.max_subsample_size)

        bp = axs.boxplot(boxplot_df,labels=[10+step_size*i for i in range(0,len(conf_list))],patch_artist=True);

        for box in bp['boxes']:
            box.set(color='#FB91A4', linewidth=2)
            box.set(facecolor = '#F5D9DD')
        for whisker in bp['whiskers']:
            whisker.set(color='#7570B3', linewidth=2)
        for cap in bp['caps']:
            cap.set(color='#7570B3', linewidth=2)
        for median in bp['medians']:
            median.set(color='#F3497F', linewidth=2)
        for flier in bp['fliers']:
            flier.set(marker='.', color='#E7298A', alpha=0.5)

        axs.plot(range(1,len(conf_list)+1),robust_and_sig, label="robust", color="#F3497F", marker="o")
        axs.plot(range(1,len(conf_list)+1),nonrobust_and_sig, label="non-robust", color="#C3A5E0", marker=".", linestyle='dashed', alpha=0.7)
        axs.plot([1,len(conf_list)+1],
                 [self.de_summary_simulation[0].shape[0],
                  self.de_summary_simulation[0].shape[0]],color="#000000", linestyle='dashed', label="total")
        axs.legend(bbox_to_anchor=(1.01, 1), loc='upper left');
        axs.set_xlabel("Number of cells subsampled per patient")
        axs.set_ylabel("Number of Signatures")
        axs.set_title("Confidence Level Estimation");
        self.confidence_de_plot = fig
        return

               
    def BarPlot(self):
        """
        Function for generating barplot for statistically significant pathways (robust and non-robust).
        Hatched bars represent signatures with statistically significant differences between groups but non-robust to subsampling.
        Solid bars represent robust signatures with statistically significant differences.
        
        """
        
        df_plot = self.de_summary[["corrected_p","U","effect","nonrobust","log2fold","direction"]]
        df_plot = df_plot.loc[df_plot.corrected_p<0.05,:]
        df_plot["log_corrected_p"] = -np.log(df_plot["corrected_p"])
        df_plot["log_corrected_p"] = [df_plot["log_corrected_p"][count] if df_plot["direction"][count] == self.treatment_group_names[0] else -1*df_plot["log_corrected_p"][count] for count in range(0,df_plot.shape[0])]
        df_plot = df_plot.sort_values(by=["log_corrected_p"],axis=0)
        size = df_plot.shape[0]

        plt.rcParams['hatch.color'] = '#FFFFFF'
        fig, axs = plt.subplots(figsize=(max(size/4,5),5))
        bar = sns.barplot(x =df_plot.index ,y= "log_corrected_p", data=df_plot, color="#6CC9B6")
        for i,thisbar in enumerate(bar.patches):

            #set different color for bars which are up in treatment-group2
            if df_plot["log_corrected_p"][i]<0:
                thisbar.set_color("#D9C5E4")

            # Set a different hatch for bars which are non-robust
            if df_plot["nonrobust"][i]:
                thisbar.set_hatch("\\")
                thisbar.set_alpha(0.5)

        plt.hlines(linestyles='dashed',y=-np.log([0.05]), xmin=-0.5, xmax=df_plot.shape[0]-0.5,colors=".3")
        axs.set_title("All statistically significant signatures")
        axs.set_ylabel("log(q)")
        axs.set_xlim(left=-0.5,right=df_plot.shape[0]-0.5)

        circ1 = mpatches.Patch(facecolor="#B2B1B0",alpha=0.5,hatch='\\\\',label='non-robust to subsampling')
        circ2 = mpatches.Patch(facecolor="#6CC9B6",label='robust to subsampling-up in '+self.treatment_group_names[0])
        circ3 = mpatches.Patch(facecolor="#D9C5E4",label='robust to subsampling-up in '+self.treatment_group_names[1])
        axs.legend(handles = [circ1,circ2,circ3],bbox_to_anchor=(1.01, 1), loc='upper left')

        plt.xticks(rotation=90);
        self.barplot = fig
        return

    
    def PatientDropoutPlot(self, annotate=True):
        """
        Function to Matrix for whether a patient 
        
        """
        non_robust_sig = self.de_summary[(self.de_summary.nonrobust==True) & (self.de_summary.corrected_p<=0.05)]
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
            pats = [x for x in fr_ratio[fr_ratio<outlier_bottom_lim[i]].index.to_list()]
            mat.iloc[i,:] = mat.columns.isin(pats).astype(int)
        
        # sort matrix accoring to frequency (rowsum, colsum)
        mat.loc["sum",] = mat.sum(axis=0)
        mat = mat.T.sort_values(by=["sum"]).T.drop(["sum"])
        
        mat["sum"] = mat.sum(axis=1)
        mat = mat.sort_values(by=["sum"]).drop(["sum"], axis=1)

        df_main = non_robust_sig_plot.loc[mat.index,mat.columns]
        
        with sns.plotting_context("notebook", rc={'axes.titlesize' : 12,
                                           'axes.labelsize' : 10,
                                           'xtick.labelsize' : 10,
                                           'ytick.labelsize' : 8,
                                           'font.name' : u'Arial'}):
            fig = plt.figure(figsize=(15,10),dpi=300)
            gs = GridSpec(4,4)
            gs.update(wspace=0.015, hspace=0.05)
            ax_main = plt.subplot(gs[1:4, :3])
            ax_yDist = plt.subplot(gs[1:4, 3], sharey=ax_main)
            ax_xDist = plt.subplot(gs[0:1, :3])

            sns.boxplot(data=df_main.T, color="#6CC9B6",ax=ax_xDist)
            ax_xDist.xaxis.set_tick_params(which='both', labeltop=False, labelbottom=False)
            ax_xDist.set_ylabel("Fold Rejection Ratio");
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
            ax_yDist.set_xlabel("#dropouts/patient");
            ax_yDist.set_xlim([0,mat.T.shape[1]]);

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
            
        self.patient_dropout_plot = fig
        return

    def HeatmapDriverGenes(self, signature_names=None, **kwargs):
        """
        Function for plotting driver genes (by default top 10) of all robust signatures.
        thought - is it possible to incorporate corr-coeff, pval as well in this plot
        
        Parameters:
            signature_names                      names of signatures for which heatmap has to be plotted
        
        """
                
        if signature_names == None:
            if self.top_signatures==None:
                print("Please run DifferentialExpression() method first")
                return
            else:
                signature_names = self.top_signatures
                    
        self.heatmap = dg.GenerateHeatmap(self.normalised_counts.T, self.t1_ids, self.t2_ids, self.d1_all, self.d2_all, self.driver_genes, signature_names, **kwargs)
        return
        
    def UpsetPlotDriverGenes(self, signature_names=None):
        """
        Function to plot intersection of driver genes between different signatures 
        (option to either enter the signature names list for which you want the plot; 
        by default will pick the top5 signatures).
        
        Parameters:
            signature_names                     (optional) names of signatures for which upsetplot should be plotted 
        
        """
                        
        if signature_names == None:
            if self.top_signatures==None:
                print("Please run DifferentialExpression() method first")
                return
            else:
                signature_names = self.top_signatures
        
        upset_df_prep = pd.DataFrame(columns=self.driver_genes.keys())
        for x in self.driver_genes.keys():
            try:
                a = self.driver_genes[x].index.to_list()
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
        upset_df = upset_df.groupby(by = signature_names).first()
        
        self.upsetplot_driver_genes = upsetplot.plot(upset_df['Intersection'], sort_by='cardinality')
        return
        
    def UpsetPlotSignatureGenes(self, signature_names=None):
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
        upset_df = upset_df.groupby(by = signature_names).first()
        self.upsetplot_signature_genes = upsetplot.plot(upset_df['Intersection'], sort_by='cardinality')
        return
    
    