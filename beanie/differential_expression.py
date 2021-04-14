"""Module for scRNA differential expression analyses."""

# built-in
import concurrent.futures
import os
import warnings

# third-party
import numpy as np
import pandas as pd
import scipy as sp

from tqdm.auto import tqdm


def mannwhitneyu(x, y, alternative: str):
    """Perform MWU."""

    try:
        result = sp.stats.mannwhitneyu(x, y, alternative=alternative)
    except ValueError:
        # catch case that arises if all input values are the same (likely all 0s)
        result = (len(x) * len(y) * 0.5, 1)
    return result


def table_mannwhitneyu(expression: pd.DataFrame, group1_cells: list, group2_cells: list, alternative='greater'):
    """Run one-sided (by default) MWU over all genes in a gene x cell expression matrix."""

    x = expression.loc[:, group1_cells].values
    y = expression.loc[:, group2_cells].values
    results = list(map(mannwhitneyu, x, y, [alternative] * len(x)))
    U = pd.Series([r[0] for r in results], index=expression.index)
    p = pd.Series([r[1] for r in results], index=expression.index)

    return U, p


def median_U_and_corresponding_p(U: np.array, p: np.array):
    """Compute median U value and get corresponding p-value from MWU U, p statistics.

    When using one-sided tests, the sort order of U & p are the same, but when using two-sided
    tests, that's not true, so sorting should be done by U alone. Complexities arise with
    even numbers of data points.

    """
    median_U = np.median(U, axis=1)

    diff_from_median = U - median_U[:, np.newaxis]
    abs_diff_from_median = np.abs(diff_from_median)

    p_below_median_U = p[(np.arange(p.shape[0]), np.apply_along_axis(np.argmin, 1, np.where(diff_from_median <= 0, abs_diff_from_median, np.inf)))]
    p_above_median_U = p[(np.arange(p.shape[0]), np.apply_along_axis(np.argmin, 1, np.where(diff_from_median >= 0, abs_diff_from_median, np.inf)))]
    p_at_median_U = (p_below_median_U + p_above_median_U) / 2

    return median_U, p_at_median_U


class DEIterationGroup:
    def __init__(self, expression: pd.DataFrame, name: str, mwu_alternative='greater'):
        self.expression = expression
        self.name = name
        self.p = None
        self.U = None
        self.effect = None
        self.cells_chosen = []
        self.group1_cell_count = None
        self.group2_cell_count = None
        self.mwu_alternative = mwu_alternative

    def run(self):
        if self.p is not None:
            # already run
            pass

        self.validate_parameters()

        iterations = len(self.cells_chosen)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(table_mannwhitneyu,
                                   [self.expression] * iterations,
                                   (x[0] for x in self.cells_chosen),
                                   (x[1] for x in self.cells_chosen),
                                   [self.mwu_alternative] * iterations,
                                   chunksize=max(1, int(iterations / os.cpu_count())))

        results = list(results)
        self.U = pd.DataFrame({f'{self.name}_{i}': r[0] for i, r in enumerate(results)})
        self.p = pd.DataFrame({f'{self.name}_{i}': r[1] for i, r in enumerate(results)})
        self.effect = self.U / (self.group1_cell_count * self.group2_cell_count)

    def validate_parameters(self):
        if len(self.cells_chosen) % 2 == 0:
            raise ValueError('Must supply odd number of iterations for well-specified median')

        self.group1_cell_count = len(self.cells_chosen[0][0])
        self.group2_cell_count = len(self.cells_chosen[0][1])
        for group1, group2 in self.cells_chosen:
            if (len(group1) != self.group1_cell_count) or (len(group2) != self.group2_cell_count):
                raise ValueError('Iterations must specify same number of cells for each comparison group')


class StandAloneDEIterationGroup(DEIterationGroup):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summary = None

    def run(self, random_seed=42):
        np.random.seed(random_seed)
        return super().run()

    def summarize(self):
        if self.summary is None:
            if self.p is None:
                self.run()

            # get p corresponding to median U value
            median_U, p_at_median_U = median_U_and_corresponding_p(self.U.values, self.p.values)

            detection_difference = pd.DataFrame(index=self.expression.index)
            for i, (group1, group2) in enumerate(self.cells_chosen):
                detection_difference[i] = (self.expression[group1] > 0).mean(axis=1) - (self.expression[group2] > 0).mean(axis=1)
            self.summary = pd.DataFrame({'p': p_at_median_U,
                                         'U': median_U,
                                         'effect': median_U / (self.group1_cell_count * self.group2_cell_count),
                                         'median_detection_difference': detection_difference.median(axis=1),
                                         'mean_detection_difference': detection_difference.mean(axis=1)}).sort_values('U')

        return self.summary


class ExcludeSampleFold(DEIterationGroup):

    def __init__(self, expression: pd.DataFrame, group1_cells: dict, group2_cells: dict,
                 group1_sample_cells: int, group2_sample_cells: int, n_subsamples: int,
                 excluded_cells: int, excluded_from_group1: bool, name: str,
                 mwu_alternative='two-sided'):
        """One fold of analyses that omit one sample per fold.
        Cells are subsampled from each non-excluded (biological) sample and differential expression
        is performed on the subsampled cells to mitigate effects of uneven cell numbers. A biological
        sample is omitted from each analyses and is replaced by cells are chosen from the sample group.

        Parameters:
            expression              gene x cell expression matrix
            group1_cells            dictionary mapping samples to cell names for samples in group 1
            group2_cells            dictionary mapping samples to cell names for samples in group 2
            group1_sample_cells     number of cells to choose from each group 1 sample
            group2_sample_cells     number of cells to choose from each group 2 sample
            n_subsamples            number of subsampling iterations
            excluded_cells          number of cells from biological sample that was excluded
            excluded_from_group1    the biological sample that was excluded is from group 1
            name                    name of fold
            mwu_alternative         alternative hypothesis for Mann Whitney U

        """
        super().__init__(expression, name, mwu_alternative=mwu_alternative)
        self.group1_cells = group1_cells
        self.group2_cells = group2_cells
        self.n_subsamples = n_subsamples
        self.group1_sample_cells = group1_sample_cells
        self.group2_sample_cells = group2_sample_cells
        self.excluded_cells = excluded_cells
        self.excluded_from_group1 = excluded_from_group1

        self.choose_cells()

    def choose_cells(self):
        """Select cells to include in each iteration."""

        if self.cells_chosen:
            # already run
            pass

        for _ in range(self.n_subsamples):
            # choose cells
            group1 = []
            group2 = []
            for cells in self.group1_cells.values():
                group1.extend(np.random.choice(cells, size=min(self.group1_sample_cells, len(cells)), replace=False))
            for cells in self.group2_cells.values():
                group2.extend(np.random.choice(cells, size=min(self.group2_sample_cells, len(cells)), replace=False))

            # make up for cells that would have been sampled from excluded sample
            replacements_needed = min(self.group1_sample_cells if self.excluded_from_group1 else self.group2_sample_cells,
                                      self.excluded_cells)
            replacement_group = self.group1_cells if self.excluded_from_group1 else self.group2_cells
            already_chosen = group1 if self.excluded_from_group1 else group2
            replacement_candidates = []
            for sample_cells in replacement_group.values():
                sample_candidates = set(sample_cells) - set(already_chosen)
                if sample_candidates:
                    replacement_candidates.append(sorted(sample_candidates))

            # weight for ~equal representation of samples
            replacement_weights = np.hstack([[1 / len(x)] * len(x) for x in replacement_candidates])
            replacement_weights /= replacement_weights.sum()
            replacement_candidates = np.hstack(replacement_candidates)
            already_chosen.extend(np.random.choice(replacement_candidates,
                                                   size=replacements_needed,
                                                   replace=False,
                                                   p=replacement_weights))

            self.cells_chosen.append((group1, group2))


class ExcludeSampleSubsampledDE:
    def __init__(self, full_expression: pd.DataFrame, group1_cells: dict, group2_cells: dict,
                 group1_sample_cells=20, group2_sample_cells=20, samples_per_fold=501,
                 min_expressing_samples=0, min_frac_per_sample=0, min_expression=0,
                 mwu_alternative='two-sided', random_state=42):
        """Perform differential expression, subsampling cells with folds dropping one sample each.

        Parameters:
            full_expression         gene x signature_score expression matrix
            group1_cells            dictionary mapping samples to cell names for samples in group 1
            group2_cells            dictionary mapping samples to cell names for samples in group 2
            group1_sample_cells     number of cells to choose from each group 1 sample
            group2_sample_cells     number of cells to choose from each group 2 sample
            samples_per_fold        number of subsampling iterations for each fold
            min_expressing_samples  minimum number of samples that express gene to be considered
            min_frac_per_sample     minimum fraction of cells expressing for a signature to be considered expressed in a sample
            min_expression          minimum expression value for a signature to be considered expressed in a cell
            mwu_alternative         alternative hypothesis for Mann Whitney U
            random_state            seed for PRNG

        """
        self.full_expression = full_expression
        self.group1_cells = group1_cells
        self.group2_cells = group2_cells
        # subset expression matrix to only genes expressed enough for consideration
        self.expression = self.full_expression.loc[
            self.expressed_genes(min_expressing_samples, min_frac_per_sample, min_expression), :]

        self.folds = []
        np.random.seed(random_state)
        exclusions = []
        if len(group1_cells) > 1:
            exclusions.extend(list(group1_cells.keys()))
        else:
            warnings.warn('Group 1 only has 1 sample, which will not be dropped in any iteration')
        if len(group2_cells) > 1:
            exclusions.extend(list(group2_cells.keys()))
        else:
            warnings.warn('Group 2 only has 1 sample, which will not be dropped in any iteration')
        if not exclusions:
            raise ValueError('Neither group has more than one sample, cannot subsample.')
        for excluded in sorted(exclusions):
            self.folds.append(ExcludeSampleFold(self.expression,
                                                {x: y for x, y in group1_cells.items() if x != excluded},
                                                {x: y for x, y in group2_cells.items() if x != excluded},
                                                group1_sample_cells,
                                                group2_sample_cells,
                                                samples_per_fold,
                                                len({**group1_cells, **group2_cells}[excluded]),
                                                excluded in group1_cells,
                                                excluded,
                                                mwu_alternative=mwu_alternative))
        self.has_run = False
        self.summary = None

    def expressed_genes(self, min_expressing_samples, min_frac_per_sample, min_expression):
        samples = list(self.group1_cells.values()) + list(self.group2_cells.values())
        expressed_samples = pd.Series(0, index=self.full_expression.index)
        for sample in samples:
            min_expressed_cells = min_frac_per_sample * len(sample)
            expressed_samples += (self.full_expression.loc[:, sample] > min_expression).sum(axis='columns') >= min_expressed_cells

        return expressed_samples[lambda x: x >= min_expressing_samples].index

    def run(self, random_seed=42):
        if not self.has_run:
            np.random.seed(random_seed)
            for fold in tqdm(self.folds):
                fold.run()

            U = pd.concat([fold.U for fold in self.folds], axis=1, sort=False).values
            p = pd.concat([fold.p for fold in self.folds], axis=1, sort=False).values
            median_U, p_at_median_U = median_U_and_corresponding_p(U, p)
            effect = median_U / (self.folds[0].group1_cell_count * self.folds[0].group2_cell_count)

            self.summary = pd.DataFrame({'p': p_at_median_U, 'U': median_U, 'effect': effect}, index=self.folds[0].U.index)
            self.has_run = True

    def summarize(self, alpha: float, min_ratio: float):
        """Return differential expression summary.
        Genes are ranked by their median U statistic. Columns after the first indicate the ratio of frequency
        that a gene was called differentially expressed in the fold that excluded the given sample
        vs the folds that included the sample. The last column indicates whether in any fold, this ratio
        was too low, indicating that the differential expression of the given gene may be driven by one
        sample only.

        Parameters:
            alpha           nominal alpha to consider a gene differentially expressed when analyzing whether
                                effects are driven by a single sample
            min_ratio       minimum ratio frequency(DE in fold excluding sample) / frequency(DE in folds including sample)

        """
        if not self.has_run:
            self.run()

        summary = self.summary.copy()
        direction = (summary['effect'] >= 0.5).values[:, np.newaxis]

        for fold in self.folds:
            fold_rejection_rate = ((fold.p.values < alpha) & ((fold.effect.values >= 0.5) == direction)).mean(axis=1)

            other_p = pd.concat([f.p for f in self.folds if f != fold], axis=1, sort=False).values
            other_direction = pd.concat([f.effect for f in self.folds if f != fold], axis=1, sort=False).values >= 0.5
            other_fold_rejection_rate = ((other_p < alpha) & (other_direction == direction)).mean(axis=1)

            summary[f'excluded_{fold.name}'] = np.divide(fold_rejection_rate, other_fold_rejection_rate)

        summary['nonrobust'] = (summary.iloc[:, 3:] < min_ratio).any(axis=1)

        return summary.sort_values(['U', 'nonrobust'])

    def serialize(self, output_dir: str):
        if not self.has_run:
            self.run()

        for fold in self.folds:
            fold.p.to_parquet(os.path.join(output_dir, f'fold_{fold.name}_p.parquet'), index=True)
            fold.U.to_parquet(os.path.join(output_dir, f'fold_{fold.name}_U.parquet'), index=True)

