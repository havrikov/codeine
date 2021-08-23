#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains tasks for the strongest support tree-based parts of the experiments.
"""

import luigi
from luigi.util import requires

from lib import tree_based, work_dir, treetools, config, generation, generation_evaluation


class ExtractStrongestSupportPathSpecification(tree_based.ExtractGenerationSpecifications):
    """Selects the strongest support path from a decision tree and translates it into a list of clauses."""

    def output(self):
        return luigi.LocalTarget(work_dir / "generation-specs" / "strongest-support" / self.format / self.subject_name)

    def select_specs(self, tree: treetools.TreeHelper):
        return [tree.strongest_support_path(config.gini_threshold)]


class StrongestSupportApproach(generation.WithGenerationMethodName):
    @property
    def generation_method_name(self) -> str:
        return "strongest-support"


@requires(ExtractStrongestSupportPathSpecification)
class PredictorBasedStrongestSupportInputGeneration(generation.PredictorBasedInputGeneration, StrongestSupportApproach):
    """Generates a single input presumably reaching the given method based on the trained predictor."""


@requires(PredictorBasedStrongestSupportInputGeneration)
class ReachMethodsWithStrongestSupportInput(generation_evaluation.EvaluateSubjectOnApproach, StrongestSupportApproach):
    """Runs the given subject with the strongest support input and produces a method coverage report."""


@requires(ReachMethodsWithStrongestSupportInput)
class ComputePrecisionForStrongestSupport(generation_evaluation.ComputePrecisionForApproach, StrongestSupportApproach):
    """Computes how often the strongest-support trees actually reach their targeted method."""
