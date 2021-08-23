#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import final

import luigi
from luigi.target import FileSystemTarget
from luigi.util import inherits
from luigi.util import requires

from lib import config
from lib import generation
from lib import generation_evaluation
from lib import tooling
from lib import tree_based
from lib import utils
from lib import work_dir


class GenerateRandomInputs(generation.RunTribble):
    """Generates random inputs with tribble."""
    output_dir: str = luigi.Parameter(description="Where to put the generated random files.", positional=False)
    number_of_files: int = luigi.IntParameter(description="How many file to generate.", positional=False)

    @property
    def generation_mode(self) -> str:
        return config.generation_mode_prefix + str(self.number_of_files)

    def output(self) -> FileSystemTarget:
        return luigi.LocalTarget(self.output_dir)


class GenerateAsManyRandomFiles(luigi.Task, utils.DynamicOutput, utils.StableRandomness, metaclass=ABCMeta):
    """Generates random sets of files to be compared against the given predictor-directed sets."""
    random_generation_seed: int = luigi.IntParameter(description="The seed for this generation run", positional=False, significant=False)

    @property
    @abstractmethod
    def comparable_task(self) -> luigi.Task:
        raise NotImplementedError("You must provide the task that produces inputs so this task can produce just as many!")

    @abstractmethod
    def output(self) -> FileSystemTarget:
        raise NotImplementedError("You must point to where random files should be generated!")

    @final
    def requires(self):
        return {
            "tribble": tooling.BuildTribble(),
            "comparison": self.comparable_task,
        }

    @final
    def run(self):
        with self.dynamic_output() as out:
            yield [self.clone(GenerateRandomInputs,
                              tribble_seed=self.random_int(self.random_generation_seed, self.format, self.subject_name, method_dir.stem),
                              output_dir=str(out / method_dir.name),
                              number_of_files=len(list(method_dir.iterdir())))
                   for method_dir in Path(self.input()["comparison"].path).iterdir()]


@inherits(tree_based.WholeTreeBasedGeneration)
class GenerateAsManyRandomFilesAsWholeTreeGeneration(GenerateAsManyRandomFiles):
    @property
    def comparable_task(self) -> luigi.Task:
        return self.clone(tree_based.WholeTreeBasedGeneration)

    def output(self):
        return luigi.LocalTarget(work_dir / "inputs" / "comparable-random" / self.format / self.subject_name)


class RandomApproach(generation.WithGenerationMethodName):
    @property
    def generation_method_name(self) -> str:
        return "random"


@requires(GenerateAsManyRandomFilesAsWholeTreeGeneration)
class ReachMethodsWithRandomInputs(generation_evaluation.EvaluateSubjectOnApproach, RandomApproach):
    """Executes a subject with random file sets and reports the covered methods."""


@requires(ReachMethodsWithRandomInputs)
class ComputePrecisionForRandom(generation_evaluation.ComputePrecisionForApproach, RandomApproach):
    """Computes how many of the randomly generated trees actually reach any given targeted method."""
