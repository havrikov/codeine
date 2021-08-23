#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains tasks for evaluating the prediction power of the learned decision trees.
"""
from pathlib import Path
from typing import final

import joblib
import luigi
import pandas as pd
from luigi.util import inherits, requires
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier

from lib import config
from lib import execution
from lib import generation
from lib import tree_based
from lib import utils
from lib import work_dir
from lib.utils import escape_method_name


class GenerateTestSet(generation.RunTribble):
    """Generates a test set for the given format to evaluate the predictors."""

    @property
    def generation_mode(self) -> str:
        return config.initial_inputs_generation_mode

    def output(self):
        return luigi.LocalTarget(work_dir / "inputs" / "test-set" / self.format)


@inherits(GenerateTestSet)
class ExtractTestSetInputFeatures(execution.ExtractInputFeatures):
    """Extracts the features of the test set."""

    @property
    def input_task(self) -> luigi.Task:
        return self.clone(GenerateTestSet)

    def output(self):
        return luigi.LocalTarget(work_dir / "input-features" / "test-set" / "raw" / f"{self.format}.csv")


@requires(ExtractTestSetInputFeatures, tree_based.FilterInvariantFeatures)
class FilterTestSetInputFeatures(luigi.Task):
    """Filters the features of the test set down to those that were also found in the training set."""

    def output(self):
        return luigi.LocalTarget(work_dir / "input-features" / "test-set" / "filtered" / f"{self.format}.csv")

    def run(self):
        # find out which features we need
        requested_features = pd.read_csv(self.input()[1].path, index_col="filename", nrows=1).columns.tolist()
        # keep only the features we want
        df = pd.read_csv(self.input()[0].path, usecols=requested_features + ["file"], index_col="file")
        df.index.rename("filename", inplace=True)
        # sort the features lexicographically
        df.sort_index(axis="columns", inplace=True)
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w", newline="") as out:
            df.to_csv(out)


@inherits(GenerateTestSet)
class RunOnTestSet(execution.RunSubject):
    """Executes a subject with the test input set and reports all covered methods."""

    @property
    def input_task(self) -> luigi.Task:
        return self.clone(GenerateTestSet)

    def output(self):
        return luigi.LocalTarget(work_dir / "evaluation" / "test-set" / self.format / "raw" / f"{self.subject_name}.csv.gz")


@final
@requires(RunOnTestSet, tree_based.TrainPredictors)
class TestSetReport(luigi.Task, utils.MethodCSVUtils, utils.CoverageOutput):
    """Transforms the coverage report as produced by the execution on test input set for further consumption."""

    def output(self):
        return luigi.LocalTarget(work_dir / "evaluation" / "test-set" / self.format / "filtered" / f"{self.subject_name}.csv.gz")

    def run(self):
        targeted_methods = set(file.stem for file in Path(self.input()[1].path).iterdir())
        # load the test set data
        test_set_methods = self.read_coverage_report(self.coverage_report(self.input()[0].path))
        # sanitize target name
        test_set_methods["target"] = test_set_methods["target"].apply(escape_method_name)
        test_set_methods = test_set_methods[test_set_methods["target"].isin(targeted_methods)]
        # index shenanigans
        test_set_methods.set_index("input_file", inplace=True)
        test_set_methods.index.rename("filename", inplace=True)
        out = Path(self.output().path)
        out.parent.mkdir(parents=True, exist_ok=True)
        test_set_methods.to_csv(out, compression="gzip")


@requires(TestSetReport, FilterTestSetInputFeatures, tree_based.TrainPredictors)
class EvaluateSubjectPredictionPower(luigi.Task):
    """Evaluates the prediction accuracy for all methods of a subject individually and compiles them into a table."""

    def output(self):
        return luigi.LocalTarget(work_dir / "results" / "accuracy-evaluation" / self.format / f"{self.subject_name}.csv")

    def run(self):
        test_set_coverage = pd.read_csv(self.input()[0].path)
        test_set_features = pd.read_csv(self.input()[1].path, index_col="filename")

        res = pd.DataFrame(columns=["subject", "target", "accuracy"])
        # for each trained classifier (i.e. each targeted method)
        for clf_path in Path(self.input()[2].path).iterdir():
            clf: DecisionTreeClassifier = joblib.load(clf_path)
            target_method = clf_path.stem
            # consider only the ground truth for the predicted method
            truth = test_set_features.merge(test_set_coverage[test_set_coverage["target"] == target_method], on="filename", validate="1:1").set_index("filename")
            y_true = truth["covered"]
            y_pred = clf.predict(truth.drop(columns=["covered", "target"]))
            # compute accuracy
            accuracy = accuracy_score(y_true, y_pred)
            # compute precision, recall, etc.
            precision, recall, f_score, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1], zero_division=1)
            # shove em in a table
            res = res.append({"subject": self.subject_name, "target": target_method,
                              "accuracy": accuracy,
                              "precision_not_covered": precision[0],
                              "precision_covered": precision[1],
                              "recall_not_covered": recall[0],
                              "recall_covered": recall[1],
                              "f_score_not_covered": f_score[0],
                              "f_score_covered": f_score[1],
                              },
                             ignore_index=True)

        with self.output().temporary_path() as out:
            res.to_csv(out, index=False)


@requires(TestSetReport, FilterTestSetInputFeatures, tree_based.TrainPredictors)
class EvaluateAveragePredictionPower(luigi.Task):
    """Evaluates the average prediction accuracy for all methods of a subject and compiles them into a table."""

    def output(self):
        return luigi.LocalTarget(work_dir / "results" / "prediction-evaluation" / self.format / f"{self.subject_name}.csv")

    @staticmethod
    def _reindex(df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index()
        df["case"] = df["filename"].str.cat(df["target"])
        df = df.drop(columns=["filename", "target"])
        return df.set_index("case")

    def run(self):
        # filename, target, covered
        df = pd.read_csv(self.input()[0].path)
        df = self._reindex(df)
        df["predicted"] = None
        # case, covered, predicted

        # filename, feature0, feature1, ...
        test_set_features = pd.read_csv(self.input()[1].path, index_col="filename")

        # for each trained classifier (i.e. each targeted method)
        for clf_path in Path(self.input()[2].path).iterdir():
            clf: DecisionTreeClassifier = joblib.load(clf_path)
            target_method = clf_path.stem
            prediction = pd.DataFrame(index=test_set_features.index)
            prediction["predicted"] = clf.predict(test_set_features)
            prediction["target"] = target_method

            df.update(self._reindex(prediction))

        # at this point it's case, covered, predicted
        df["predicted"] = df["predicted"].astype(int)

        y_true = df["covered"]
        y_pred = df["predicted"]
        # compute accuracy
        accuracy = accuracy_score(y_true, y_pred)
        # compute precision, recall, etc.
        precision, recall, f_score, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

        res = pd.DataFrame(data={
            "format": [self.format],
            "subject": [self.subject_name],
            "accuracy": [accuracy],
            "precision": [precision],
            "recall": [recall],
            "f_score": [f_score],
        })

        with self.output().temporary_path() as out:
            res.to_csv(out, index=False)
