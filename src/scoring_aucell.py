# from ctxcore.recovery import enrichment4cells
from typing import Sequence, Type
from .genesig import GeneSignature
from multiprocessing import cpu_count, Process, Array
from boltons.iterutils import chunked
from multiprocessing.sharedctypes import RawArray
from operator import mul
import logging
from math import ceil
from ctypes import c_uint32
from operator import attrgetter
import pandas as pd
import numpy as np
from itertools import repeat
from typing import List, Optional, Tuple

LOGGER = logging.getLogger(__name__)
# To reduce the memory footprint of a ranking matrix we use unsigned 32bit integers which provides a range from 0
# through 4,294,967,295. This should be sufficient even for region-based approaches.
DTYPE = 'uint32'
DTYPE_C = c_uint32

def derive_rank_cutoff(
    auc_threshold: float, total_genes: int, rank_threshold: Optional[int] = None
) -> int:
    """
    Get rank cutoff.
    :param auc_threshold: The fraction of the ranked genome to take into account for
        the calculation of the Area Under the recovery Curve.
    :param total_genes: The total number of genes ranked.
    :param rank_threshold: The total number of ranked genes to take into account when
        creating a recovery curve.
    :return Rank cutoff.
    """

    if not rank_threshold:
#         print("here")
        rank_threshold = total_genes - 1

    assert (
        0 < rank_threshold < total_genes
    ), f"Rank threshold must be an integer between 1 and {total_genes:d}."
    assert (
        0.0 < auc_threshold <= 1.0
    ), "AUC threshold must be a fraction between 0.0 and 1.0."

    # In the R implementation the cutoff is rounded.
    rank_cutoff = int(round(auc_threshold * total_genes))
    assert 0 < rank_cutoff <= rank_threshold, (
        f"An AUC threshold of {auc_threshold:f} corresponds to {rank_cutoff:d} top "
        f"ranked genes/regions in the database. Please increase the rank threshold "
        "or decrease the AUC threshold."
    )
    # Make sure we have exactly the same AUC values as the R-SCENIC pipeline.
    # In the latter the rank threshold is not included in AUC calculation.
    rank_cutoff -= 1
    return rank_cutoff

def weighted_auc1d(
    ranking: np.ndarray, weights: np.ndarray, rank_cutoff: int, max_auc: float
) -> np.ndarray:
    """
    Calculate the AUC of the weighted recovery curve of a single ranking.
    :param ranking: The rank numbers of the genes.
    :param weights: The associated weights.
    :param rank_cutoff: The maximum rank to take into account when calculating the AUC.
    :param max_auc: The maximum AUC.
    :return: The normalized AUC.
    """
    # Using concatenate and full constructs required by numba.
    # The rankings are 0-based. The position at the rank threshold is included in the calculation.
    filter_idx = ranking < rank_cutoff
    x = ranking[filter_idx]
    y = weights[filter_idx]
    sort_idx = np.argsort(x)
    x = np.concatenate((x[sort_idx], np.full((1,), rank_cutoff, dtype=np.int_)))
    y = y[sort_idx].cumsum()
    return np.sum(np.diff(x) * y) / max_auc

def auc2d(
    rankings: np.ndarray, weights: np.ndarray, rank_cutoff: int, max_auc: float
) -> np.ndarray:
    """
    Calculate the AUCs of multiple rankings.
    :param rankings: The rankings.
    :param weights: The weights associated with the selected genes.
    :param rank_cutoff: The maximum rank to take into account when calculating the AUC.
    :param max_auc: The maximum AUC.
    :return: The normalized AUCs.
    """
    n_features = rankings.shape[0]
    aucs = np.empty(shape=(n_features,), dtype=np.float64)  # Pre-allocation.
    for row_idx in range(n_features):
        aucs[row_idx] = weighted_auc1d(
            rankings[row_idx, :], weights, rank_cutoff, max_auc
        )
    return aucs


def aucs(
    rnk: pd.DataFrame, total_genes: int, weights: np.ndarray, auc_threshold: float
) -> np.ndarray:
    """
    Calculate AUCs (implementation without calculating recovery curves first).
    :param rnk: A dataframe containing the rank number of genes of interest. Columns correspond to genes.
    :param total_genes: The total number of genes ranked.
    :param weights: The weights associated with the selected genes.
    :param auc_threshold: The fraction of the ranked genome to take into account for the calculation of the
        Area Under the recovery Curve.
    :return: An array with the AUCs.
    """
    rank_cutoff = derive_rank_cutoff(auc_threshold, total_genes)
    _features, _genes, rankings = rnk.index.values, rnk.columns.values, rnk.values
    y_max = weights.sum()
    # The rankings are 0-based. The position at the rank threshold is included in the calculation.
    # The maximum AUC takes this into account.
    # For reason of generating the same results as in R we introduce an error by adding one to the rank_cutoff
    # for calculationg the maximum AUC.
    maxauc = float((rank_cutoff + 1) * y_max)
    assert maxauc > 0
    return auc2d(rankings, weights, rank_cutoff, maxauc)


def enrichment4cells(
    rnk_mtx: pd.DataFrame, regulon: GeneSignature, auc_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Calculate the enrichment of the regulon for the cells in the ranking dataframe.
    :param rnk_mtx: The ranked expression matrix (n_cells, n_genes).
    :param regulon: The regulon the assess for enrichment
    :param auc_threshold: The fraction of the ranked genome to take into account for the calculation of the
        Area Under the recovery Curve.
    :return:
    """

    total_genes = len(rnk_mtx.columns)
    index = pd.MultiIndex.from_tuples(
        list(zip(rnk_mtx.index.values, repeat(regulon.name))), names=["Cell", "Regulon"]
    )
    rnk = rnk_mtx.iloc[:, rnk_mtx.columns.isin(regulon.genes)]
    if rnk.empty or (float(len(rnk.columns)) / float(len(regulon))) < 0.80:
        LOGGER.warning(
            f"Less than 80% of the genes in {regulon.name} are present in the "
            "expression matrix."
        )
        return pd.DataFrame(
            index=index,
            data={"AUC": np.zeros(shape=(rnk_mtx.shape[0]), dtype=np.float64)},
        )
    else:
        weights = np.asarray(
            [
                regulon[gene] if gene in regulon.genes else 1.0
                for gene in rnk.columns.values
            ]
        )
        return pd.DataFrame(
            index=index, data={"AUC": aucs(rnk, total_genes, weights, auc_threshold)}
        )

def create_rankings(ex_mtx: pd.DataFrame, seed=None) -> pd.DataFrame:
    """
    Create a whole genome rankings dataframe from a single cell expression profile dataframe.
    :param ex_mtx: The expression profile matrix. The rows should correspond to different cells, the columns to different
        genes (n_cells x n_genes).
    :return: A genome rankings dataframe (n_cells x n_genes).
    """
    # Do a shuffle would be nice for exactly similar behaviour as R implementation.
    # 1. Ranks are assigned in the range of 1 to n, therefore we need to subtract 1.
    # 2. In case of a tie the 'first' method is used, i.e. we keep the order in the original array. The remove any
    #    bias we shuffle the dataframe before ranking it. This introduces a performance penalty!
    # 3. Genes are ranked according to gene expression in descending order, i.e. from highly expressed (0) to low expression (n).
    # 3. NAs should be given the highest rank numbers. Documentation is bad, so tested implementation via code snippet:
    #
    #    import pandas as pd
    #    import numpy as np
    #    df = pd.DataFrame(data=[4, 1, 3, np.nan, 2, 3], columns=['values'])
    #    # Run below statement multiple times to see effect of shuffling in case of a tie.
    #    df.sample(frac=1.0, replace=False).rank(ascending=False, method='first', na_option='bottom').sort_index() - 1
    #
    return (
        ex_mtx.sample(frac=1.0, 
                 replace=False, axis=1, 
                 random_state=3120).rank(axis=0, 
                                         pct=True,
                                         ascending=True, 
                                         method='first', 
                                         na_option='bottom').rank(axis=1,
                                                                  ascending=False, 
                                                                  method='first', 
                                                                  na_option='bottom').astype(DTYPE) - 1
    )


def derive_auc_threshold(ex_mtx: pd.DataFrame) -> pd.DataFrame:
    """
    Derive AUC thresholds for an expression matrix.
    It is important to check that most cells have a substantial fraction of expressed/detected genes in the calculation of
    the AUC.
    :param ex_mtx: The expression profile matrix. The rows should correspond to different cells, the columns to different
        genes (n_cells x n_genes).
    :return: A dataframe with AUC threshold for different quantiles over the number cells: a fraction of 0.01 designates
        that when using this value as the AUC threshold for 99% of the cells all ranked genes used for AUC calculation will
        have had a detected expression in the single-cell experiment.
    """
    return pd.Series(np.count_nonzero(ex_mtx, axis=1)).quantile([0.01, 0.05, 0.10, 0.50, 1]) / ex_mtx.shape[1]


enrichment = enrichment4cells


def _enrichment(shared_ro_memory_array, modules, genes, cells, auc_threshold, auc_mtx, offset):
    # The rankings dataframe is properly reconstructed (checked this).
    df_rnk = pd.DataFrame(
        data=np.frombuffer(shared_ro_memory_array, dtype=DTYPE).reshape(len(cells), len(genes)),
        columns=genes,
        index=cells,
    )
    # To avoid additional memory burden de resulting AUCs are immediately stored in the output sync. array.
    result_mtx = np.frombuffer(auc_mtx.get_obj(), dtype='d')
    inc = len(cells)
    for idx, module in enumerate(modules):
        result_mtx[offset + (idx * inc) : offset + ((idx + 1) * inc)] = enrichment4cells(
            df_rnk, module, auc_threshold
        ).values.flatten(order="C")


def aucell4r(
    df_rnk: pd.DataFrame,
    signatures: Sequence[Type[GeneSignature]],
    auc_threshold: float = 0.05,
    noweights: bool = False,
    normalize: bool = False,
    num_workers: int = cpu_count(),
) -> pd.DataFrame:
    """
    Calculate enrichment of gene signatures for single cells.
    :param df_rnk: The rank matrix (n_cells x n_genes).
    :param signatures: The gene signatures or regulons.
    :param auc_threshold: The fraction of the ranked genome to take into account for the calculation of the
        Area Under the recovery Curve.
    :param noweights: Should the weights of the genes part of a signature be used in calculation of enrichment?
    :param normalize: Normalize the AUC values to a maximum of 1.0 per regulon.
    :param num_workers: The number of cores to use.
    :return: A dataframe with the AUCs (n_cells x n_modules).
    """
    if num_workers == 1:
        # Show progress bar ...
        aucs = pd.concat(
            [
                enrichment4cells(df_rnk, module.noweights() if noweights else module, auc_threshold=auc_threshold)
                for module in tqdm(signatures)
            ]
        ).unstack("Regulon")
        aucs.columns = aucs.columns.droplevel(0)
    else:
        # Decompose the rankings dataframe: the index and columns are shared with the child processes via pickling.
        genes = df_rnk.columns.values
        cells = df_rnk.index.values
        # The actual rankings are shared directly. This is possible because during a fork from a parent process the child
        # process inherits the memory of the parent process. A RawArray is used instead of a synchronize Array because
        # these rankings are read-only.
        shared_ro_memory_array = RawArray(DTYPE_C, mul(*df_rnk.shape))
        array = np.frombuffer(shared_ro_memory_array, dtype=DTYPE)
        # Copy the contents of df_rank into this shared memory block using row-major ordering.
        array[:] = df_rnk.values.flatten(order='C')

        # The resulting AUCs are returned via a synchronize array.
        auc_mtx = Array('d', len(cells) * len(signatures))  # Double precision floats.

        # Convert the modules to modules with uniform weights if necessary.
        if noweights:
            signatures = list(map(lambda m: m.noweights(), signatures))

        # Do the analysis in separate child processes.
        chunk_size = ceil(float(len(signatures)) / num_workers)
        processes = [
            Process(
                target=_enrichment,
                args=(
                    shared_ro_memory_array,
                    chunk,
                    genes,
                    cells,
                    auc_threshold,
                    auc_mtx,
                    (chunk_size * len(cells)) * idx,
                ),
            )
            for idx, chunk in enumerate(chunked(signatures, chunk_size))
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        # Reconstitute the results array. Using C or row-major ordering.
        aucs = pd.DataFrame(
            data=np.ctypeslib.as_array(auc_mtx.get_obj()).reshape(len(signatures), len(cells)),
            columns=pd.Index(data=cells, name='Cell'),
            index=pd.Index(data=list(map(attrgetter("name"), signatures)), name='Regulon'),
        ).T
    return aucs / aucs.max(axis=0) if normalize else aucs


def aucell(
    rnk_mtx: pd.DataFrame,
    signatures: Sequence[Type[GeneSignature]],
    auc_threshold: float = 0.05,
    noweights: bool = False,
    normalize: bool = False,
    seed=None,
    num_workers: int = cpu_count(),
) -> pd.DataFrame:
    """
    Calculate enrichment of gene signatures for single cells.
    :param exp_mtx: The expression matrix (n_cells x n_genes).
    :param signatures: The gene signatures or regulons.
    :param auc_threshold: The fraction of the ranked genome to take into account for the calculation of the
        Area Under the recovery Curve.
    :param noweights: Should the weights of the genes part of a signature be used in calculation of enrichment?
    :param normalize: Normalize the AUC values to a maximum of 1.0 per regulon.
    :param num_workers: The number of cores to use.
    :return: A dataframe with the AUCs (n_cells x n_modules).
    """
    return aucell4r(rnk_mtx, signatures, auc_threshold, noweights, normalize, num_workers)