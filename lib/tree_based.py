#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains tasks for the tree-based parts of the experiments.
"""

import random
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List, Iterable, Dict, final

import joblib
import logging
import luigi
import pandas as pd
from luigi.target import FileSystemTarget
from luigi.util import requires, inherits
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier

from lib import config
from lib import execution
from lib import generation
from lib import generation_evaluation
from lib import treetools
from lib import utils
from lib import work_dir
from lib.utils import escape_method_name, _percentage_to_variance


class GenerateInitialInputs(generation.RunTribble):
    """Generates initial inputs with tribble."""

    @property
    def generation_mode(self) -> str:
        return config.initial_inputs_generation_mode

    def output(self):
        return luigi.LocalTarget(work_dir / "inputs" / "initial" / self.format)


@inherits(GenerateInitialInputs)
class ExtractInitialInputFeatures(execution.ExtractInputFeatures):
    """Parses the inputs and extracts input features."""

    @property
    def input_task(self) -> luigi.Task:
        return self.clone(GenerateInitialInputs)

    def output(self):
        return luigi.LocalTarget(work_dir / "input-features" / "initial" / "raw" / f"{self.format}.csv")


@requires(ExtractInitialInputFeatures)
class FilterInvariantFeatures(luigi.Task):
    """Filters out invariant input features."""

    def output(self):
        return luigi.LocalTarget(work_dir / "input-features" / "initial" / "filtered" / f"{self.format}.csv")

    def run(self):
        # the csv should be structured like this: "filename", "feature1", "feature2", ..., "featureN"
        # but right now it still has a lot of additional columns (and filename is named file)
        df = pd.read_csv(self.input().path, index_col="file")
        # rename the index from "file" to "filename" for compatibility with the method features csv ;)
        df.index.rename("filename", inplace=True)
        # only keep feature columns
        input_features = df.drop(columns=["status", "abs_file"])
        # now we can filter out invariant features
        selector = VarianceThreshold(_percentage_to_variance(config.feature_variance_threshold))
        X = selector.fit_transform(input_features)
        feature_labels = input_features.columns[selector.get_support(True)]
        X = pd.DataFrame(data=X, columns=feature_labels, index=input_features.index)
        # sort the features lexicographically
        X.sort_index(axis="columns", inplace=True)
        with self.output().temporary_path() as out:
            X.to_csv(out)


@inherits(GenerateInitialInputs)
class RunSubjectOnInitialInputs(execution.RunSubject):
    """Runs the given subject with the initial inputs and produces a method coverage report."""

    @property
    def input_task(self) -> luigi.Task:
        return self.clone(GenerateInitialInputs)

    def output(self):
        return luigi.LocalTarget(work_dir / "evaluation" / "initial" / "raw" / self.format / f"{self.subject_name}.csv.gz")


@requires(RunSubjectOnInitialInputs)
class FilterInvariantMethods(luigi.Task, utils.MethodCSVUtils, utils.CoverageOutput):
    """Filters out methods that were executed or not executed in all runs."""
    subsampling_seed: int = luigi.IntParameter(description="The seed to use if subsampling is chosen", positional=False, significant=False)

    def output(self):
        return luigi.LocalTarget(work_dir / "evaluation" / "initial" / "filtered" / self.format / f"{self.subject_name}.csv.gz")

    @staticmethod
    def _filter_by_variance_threshold(methods_as_features):
        selector = VarianceThreshold(_percentage_to_variance(config.method_variance_threshold))
        df = selector.fit_transform(methods_as_features)
        labels = methods_as_features.columns[selector.get_support(True)]
        df = pd.DataFrame(data=df, columns=labels, index=methods_as_features.index)
        return df

    @staticmethod
    def _preserve_selection(col: pd.Series, rnd, k: int) -> None:
        s = len(col) - k
        selected = rnd.sample(list(col.index), s)
        col[selected] = None

    def _subselect(self, rnd: random.Random, col: pd.Series) -> None:
        zeros = col[col == 0]
        ones = col[col == 1]
        k = min(len(zeros), len(ones))
        self._preserve_selection(zeros, rnd, k)
        if not config.subsample_bias:
            self._preserve_selection(ones, rnd, k)

    def _subsample(self, df: pd.DataFrame) -> None:
        rnd = random.Random(self.subsampling_seed)
        for col in df.columns:
            self._subselect(rnd, df[col])

    def run(self):
        # the gzipped csv is structured like this: "input_file", "class_name", "method_name", "line", "instructions_missed", "instructions_hit"
        methods = self.read_coverage_report(self.coverage_report(self.input().path))
        # pivot the dataframe such that it can be used with ML
        methods_as_features = methods.pivot(index="input_file", columns="target", values="covered")
        # rename the index from "input_file" to "filename" for compatibility with the input features csv
        methods_as_features.index.rename("filename", inplace=True)
        # filter out methods that are covered or not covered in all of the inputs
        # because it is impossible to know which features are responsible for covering or not covering them
        df = self._filter_by_variance_threshold(methods_as_features)
        if config.use_subsampling:
            self._subsample(df)
        out = Path(self.output().path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, compression="gzip")


@requires(FilterInvariantFeatures, FilterInvariantMethods)
class SelectMethods(luigi.Task, utils.StableRandomness):
    """Randomly selects method targets for which to train predictors."""
    selection_seed: int = luigi.IntParameter(description="The seed for the selection of the method targets", positional=False)

    def output(self):
        return luigi.LocalTarget(work_dir / "training-data" / self.format / self.subject_name)

    def run(self):
        input_features = pd.read_csv(self.input()[0].path, index_col="filename")
        all_methods = pd.read_csv(self.input()[1].path, index_col="filename")
        rnd = self.get_random(self.selection_seed, self.format, self.subject_name)
        with self.output().temporary_path() as out:
            out_dir = Path(out)
            out_dir.mkdir(parents=True)
            all_method_names: List[str] = all_methods.columns.tolist()
            selection = rnd.sample(all_method_names, k=min(config.number_of_methods, len(all_method_names)))
            for column in selection:
                df = pd.merge(input_features, all_methods[column], on="filename", validate="1:1").dropna()
                df.to_csv(out_dir / f"{escape_method_name(column)}.csv")


@requires(SelectMethods)
class TrainPredictors(luigi.Task, utils.StableRandomness):
    """Trains trees for each selected method for a given subject."""
    training_seed: int = luigi.IntParameter(description="The seed for training the decision tree classifiers", positional=False)

    def output(self):
        return luigi.LocalTarget(work_dir / "predictors" / self.format / self.subject_name)

    def run(self):
        with self.output().temporary_path() as tmp:
            out_dir = Path(tmp)
            out_dir.mkdir(parents=True)
            for method in Path(self.input().path).iterdir():
                tree_seed = self.random_int(self.training_seed, self.format, self.subject_name, method.stem)
                df = pd.read_csv(method, index_col="filename")
                min_leaf = config.min_samples_leaf
                if min_leaf.is_integer():
                    min_leaf = int(min_leaf)
                clf = DecisionTreeClassifier(max_depth=config.max_tree_depth, random_state=tree_seed, min_samples_leaf=min_leaf)
                clf = clf.fit(df.iloc[:, 0:-1], df.iloc[:, -1])
                joblib.dump(clf, out_dir / method.with_suffix(".clf").name)


@requires(FilterInvariantFeatures, TrainPredictors)
class ExtractGenerationSpecifications(luigi.Task, metaclass=ABCMeta):
    """Translates decision trees into lists of derivation rules to include or exclude."""

    @abstractmethod
    def output(self) -> FileSystemTarget:
        raise NotImplementedError("You must point to where to put the spec files!")

    @abstractmethod
    def select_specs(self, tree: treetools.TreeHelper) -> Iterable[Dict[str, bool]]:
        return tree.covering_specifications(config.gini_threshold)

    @final
    def run(self):
        feature_names = pd.read_csv(self.input()[0].path, index_col="filename", nrows=1).columns.tolist()
        with self.output().temporary_path() as tmp:
            out_dir = Path(tmp)
            out_dir.mkdir(parents=True)
            for tree_path in Path(self.input()[1].path).iterdir():
                clf: DecisionTreeClassifier = joblib.load(tree_path)
                tree = treetools.TreeHelper(clf, feature_names)
                # create a spec file with a line for each path in the classifier tree
                specs = [self._specs_to_str(spec) for spec in self.select_specs(tree)]
                if not specs:
                    logging.warning(f"Empty tree for {tree_path.stem}!")
                    continue
                outfile = out_dir / tree_path.with_suffix(".spec").name
                with outfile.open("w") as out:
                    out.writelines(specs)

    @staticmethod
    @final
    def _specs_to_str(specs: Dict[str, bool]) -> str:
        return " ".join((("" if v else "!") + k for k, v in specs.items())) + "\n"


class ExtractWholeTreeGenerationSpecifications(ExtractGenerationSpecifications):
    """Translates the whole decision trees into lists of derivation rules to include or exclude."""

    def output(self):
        return luigi.LocalTarget(work_dir / "generation-specs" / "whole-tree" / self.format / self.subject_name)

    def select_specs(self, tree: treetools.TreeHelper):
        return tree.covering_specifications(config.gini_threshold)


class WholeTreeApproach(generation.WithGenerationMethodName):
    @property
    def generation_method_name(self) -> str:
        return "whole-tree"


@requires(ExtractWholeTreeGenerationSpecifications)
class WholeTreeBasedGeneration(generation.PredictorBasedInputGeneration, WholeTreeApproach):
    """Generates inputs for the specs extracted from whole decision trees."""


@requires(WholeTreeBasedGeneration)
class ReachMethodsWithWholeTreeInputs(generation_evaluation.EvaluateSubjectOnApproach, WholeTreeApproach):
    """Runs the given subject with whole-tree-generated inputs and produces a method coverage report for each method."""


@requires(ReachMethodsWithWholeTreeInputs)
class ComputePrecisionForWholeTree(generation_evaluation.ComputePrecisionForApproach, WholeTreeApproach):
    """Computes how many of the whole-tree-generated trees actually reach their targeted method."""


@requires(
    RunSubjectOnInitialInputs,
    FilterInvariantMethods,
    SelectMethods,
    WholeTreeBasedGeneration,
    ComputePrecisionForWholeTree,
)
class CreateMethodNumberReport(luigi.Task, utils.MethodCSVUtils, utils.CoverageOutput):
    """Reports the number of methods and what we could do with them for a given subject."""

    def output(self):
        return luigi.LocalTarget(work_dir / "results" / "method-numbers" / self.format / f"{self.subject_name}.csv")

    def run(self):
        # observed
        observed = self.read_coverage_report(self.coverage_report(self.input()[0].path))
        num_observed = observed["target"].nunique()

        # methods with variance
        num_trainable = pd.read_csv(self.input()[1].path, nrows=1).shape[1] - 1

        # methods that were selected
        num_selected = len(list(Path(self.input()[2].path).iterdir()))

        # number of methods with successfully generated inputs
        num_generated = len([d for d in Path(self.input()[3].path).iterdir() if any(d.iterdir())])

        # number of actually reached methods
        reached = pd.read_csv(self.input()[4].path)
        num_reached = len(reached[reached["covered"] > 0])

        res = pd.DataFrame(data={
            "format": [self.format],
            "subject": [self.subject_name],
            "observed": [num_observed],
            "trainable": [num_trainable],
            "selected": [num_selected],
            "generated": [num_generated],
            "reached": [num_reached],
        })

        with self.output().temporary_path() as out:
            res.to_csv(out, index=False)
