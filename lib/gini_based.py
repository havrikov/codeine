#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains tasks for the lowest gini tree-based parts of the experiments.
"""

import luigi
from luigi.util import requires

from lib import tree_based, work_dir, treetools, generation, generation_evaluation


class ExtractLowestGiniPathSpecification(tree_based.ExtractGenerationSpecifications):
    """Selects the strongest gini path from a decision tree and translates it into a list of clauses."""

    def output(self):
        return luigi.LocalTarget(work_dir / "generation-specs" / "lowest-gini" / self.format / self.subject_name)

    def select_specs(self, tree: treetools.TreeHelper):
        return [tree.strongest_gini_path()]


class LowestGiniApproach(generation.WithGenerationMethodName):
    @property
    def generation_method_name(self) -> str:
        return "lowest-gini"


@requires(ExtractLowestGiniPathSpecification)
class PredictorBasedLowestGiniInputGeneration(generation.PredictorBasedInputGeneration, LowestGiniApproach):
    """Generates a single input presumably reaching the given method based on the trained predictor."""


@requires(PredictorBasedLowestGiniInputGeneration)
class ReachMethodWithLowestGiniInput(generation_evaluation.EvaluateSubjectOnApproach, LowestGiniApproach):
    """Runs the given subject with the strongest gini input and produces a method coverage report."""


@requires(ReachMethodWithLowestGiniInput)
class ComputePrecisionForLowestGini(generation_evaluation.ComputePrecisionForApproach, LowestGiniApproach):
    """Computes how often the lowest-gini trees actually reach their targeted method."""
