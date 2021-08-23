#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains luigi tasks for evaluating the method execution counts.
"""
from pathlib import Path
from statistics import mean, median

import luigi
import pandas as pd
from luigi.util import requires

from lib import tree_based, work_dir, prediction_evaluation, utils, random_based


@requires(prediction_evaluation.RunOnTestSet, tree_based.ReachMethodsWithWholeTreeInputs)
class EvaluateConstrainedness(luigi.Task, utils.CoverageOutput, utils.MethodCSVUtils):
    """Evaluates how many methods are covered on average by random and tree-based approaches."""

    def output(self):
        return luigi.LocalTarget(work_dir / "results" / "constrainedness" / self.format / f"{self.subject_name}.csv")

    @staticmethod
    def _covered_method_counts(df: pd.DataFrame) -> pd.Series:
        # keep only covered
        df = df[df["covered"] == 1].drop(columns="target")
        # for each file count the methods
        counts = df.groupby("input_file").count()
        return counts["covered"]

    def run(self):
        # take ground truth and calculate the mean and median
        truth = self.read_coverage_report(self.coverage_report(self.input()[0].path))
        truth_counts = self._covered_method_counts(truth)
        rnd_mean = truth_counts.mean()
        rnd_median = truth_counts.median()

        # gather tree-generated inputs and calculate their average
        tree_generated = Path(self.input()[1].path) / self.coverage_dir_name
        dfs = []
        for file in tree_generated.iterdir():
            df = self.read_coverage_report(file)
            df["input_file"] = file.name + df["input_file"].astype(str)
            dfs.append(df)
        cumulative_tree_based = pd.concat(dfs)
        tree_counts = self._covered_method_counts(cumulative_tree_based)
        tree_mean = tree_counts.mean()
        tree_median = tree_counts.median()

        df = pd.DataFrame(data={
            "format": [self.format],
            "subject": [self.subject_name],
            "average # methods reached with random": [rnd_mean],
            "average # methods reached with whole-tree": [tree_mean],
            "median # methods reached with random": [rnd_median],
            "median # methods reached with whole-tree": [tree_median],
        })

        with self.output().temporary_path() as out:
            df.to_csv(out, index=False)


def _wilcoxon_diff_report(whole_tree_counts, random_counts, fmt: str, subject_name: str) -> pd.DataFrame:
    diffs = [a - b for a, b in zip(whole_tree_counts, random_counts)]
    w_two_sided, p_two_sided = utils.safe_wilcoxon(diffs, alternative="two-sided")
    w_greater, p_greater = utils.safe_wilcoxon(diffs, alternative="greater")
    return pd.DataFrame(data={
        "format": [fmt],
        "subject": [subject_name],
        "mean difference": [(mean(diffs))],
        "median difference": [(median(diffs))],
        "min difference": [(min(diffs))],
        "max difference": [(max(diffs))],
        "wilcoxon (two-sided)": [w_two_sided],
        "p-value (two-sided)": [p_two_sided],
        "wilcoxon (greater)": [w_greater],
        "p-value (greater)": [p_greater],
    })


@requires(tree_based.ReachMethodsWithWholeTreeInputs, random_based.ReachMethodsWithRandomInputs, tree_based.SelectMethods)
class EvaluateReachedMethodCount(luigi.Task, utils.MethodCSVUtils, utils.CoverageOutput):
    """Evaluates how many methods are reached in general by random and tree-based approaches."""

    def output(self):
        return luigi.LocalTarget(work_dir / "results" / "method-counts" / self.format / f"{self.subject_name}.csv")

    def _unique_methods_covered(self, path: Path) -> int:
        df = self.read_coverage_report(path)
        df = df[df["covered"] == 1]
        return df["target"].nunique()

    def run(self):
        whole_tree_counts = []
        random_counts = []
        for target_path in Path(self.input()[2].path).iterdir():
            target_method = target_path.stem
            whole_tree_path = self.coverage_report(Path(self.input()[0].path) / f"{target_method}.csv.gz")
            # skip methods with unproductive trees
            if not whole_tree_path.exists():
                continue
            whole_tree_counts.append(self._unique_methods_covered(whole_tree_path))

            random_path = self.coverage_report(Path(self.input()[1].path) / f"{target_method}.csv.gz")
            random_counts.append(self._unique_methods_covered(random_path))

        df = _wilcoxon_diff_report(whole_tree_counts, random_counts, self.format, self.subject_name)

        with self.output().temporary_path() as out:
            df.to_csv(out, index=False)


@requires(tree_based.ReachMethodsWithWholeTreeInputs, random_based.ReachMethodsWithRandomInputs, tree_based.SelectMethods)
class EvaluateTargetMethodExecutionCount(luigi.Task, utils.CoverageOutput):
    """Reports the signed wilcoxon rank test results comparing random and tree-based approaches."""

    def output(self):
        return luigi.LocalTarget(work_dir / "results" / "target-method-stats" / self.format / f"{self.subject_name}.csv")

    def run(self):
        whole_tree_counts = []
        random_counts = []
        for target_path in Path(self.input()[2].path).iterdir():
            target_method = target_path.stem
            whole_tree_hitcounts = self.read_hitcounts(Path(self.input()[0].path) / target_method)
            random_hitcounts = self.read_hitcounts(Path(self.input()[1].path) / target_method)

            whole_tree_counts.append(self._target_hitcount(whole_tree_hitcounts, target_method))
            random_counts.append(self._target_hitcount(random_hitcounts, target_method))

        df = _wilcoxon_diff_report(whole_tree_counts, random_counts, self.format, self.subject_name)

        with self.output().temporary_path() as out:
            df.to_csv(out, index=False)

    @staticmethod
    def _target_hitcount(df: pd.DataFrame, method: str) -> int:
        """Get the hitcount of the given method from the given data frame or zero if it is absent."""
        try:
            return df.loc[method].squeeze()
        except KeyError:
            return 0


@requires(tree_based.ReachMethodsWithWholeTreeInputs, random_based.ReachMethodsWithRandomInputs, tree_based.SelectMethods)
class CollateBycatchExecutionCounts(luigi.Task, utils.CoverageOutput):
    """Collates the execution counts of non-targeted methods for random and tree-based approaches for the given subject."""

    def output(self):
        return luigi.LocalTarget(work_dir / "results" / "bycatch-counts" / self.format / f"{self.subject_name}.csv")

    def run(self):
        dfs = []
        for target_path in Path(self.input()[2].path).iterdir():
            target_method = target_path.stem
            whole_tree_hitcounts = self.read_hitcounts(Path(self.input()[0].path) / target_method)
            random_hitcounts = self.read_hitcounts(Path(self.input()[1].path) / target_method)
            dj = whole_tree_hitcounts.join(random_hitcounts, how="outer", lsuffix="_tree", rsuffix="_random").fillna(0)
            dj["targeted-method"] = target_method
            dj["observed-method"] = dj.index
            dj = dj[dj["observed-method"] != target_method]
            dj.set_index(["targeted-method", "observed-method"], inplace=True)
            dfs.append(dj)

        df = pd.concat(dfs)
        df = df.astype({"executions_tree": int, "executions_random": int})
        df["format"] = self.format
        df["subject"] = self.subject_name

        with self.output().temporary_path() as out:
            df.to_csv(out, index=True)


@requires(CollateBycatchExecutionCounts)
class EvaluateBycatchExecutionCount(luigi.Task):
    """Compares the execution counts of non-targeted methods for random and tree-based approaches for the given subject."""

    def output(self):
        return luigi.LocalTarget(work_dir / "results" / "bycatch-stats" / self.format / f"{self.subject_name}.csv")

    def run(self):
        df = pd.read_csv(self.input().path, index_col=["targeted-method", "observed-method"])
        res = _wilcoxon_diff_report(df["executions_tree"], df["executions_random"], self.format, self.subject_name)

        with self.output().temporary_path() as out:
            res.to_csv(out, index=False)
