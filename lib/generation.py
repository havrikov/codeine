#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains luigi tasks for running tasks that generate input files.
"""

import logging
import subprocess
from abc import ABCMeta, ABC
from abc import abstractmethod
from pathlib import Path
from typing import final

import luigi
from luigi.target import FileSystemTarget
from luigi.util import requires

from lib import config
from lib import subjects
from lib import tooling
from lib import utils
from lib import work_dir


class RunTribble(luigi.Task, utils.StableRandomness, metaclass=ABCMeta):
    """Base class for creating inputs with tribble."""
    format: str = luigi.Parameter(description="The name of the format directory (e.g. json)", positional=False)
    tribble_seed: int = luigi.IntParameter(description="The seed for this tribble run", positional=False, significant=False)
    resources = {"ram": 4}

    @property
    @abstractmethod
    # This is a property and not a parameter so that @requires can be used without propagating this further up the dependency tree.
    def generation_mode(self) -> str:
        raise NotImplementedError("You must provide the tribble generation mode to use!")

    @abstractmethod
    def output(self) -> FileSystemTarget:
        raise NotImplementedError("You must point to where to put the generated files!")

    @final
    def requires(self):
        d = {"tribble": tooling.BuildTribble()}
        if config.use_grammar_caching:
            d["cache"] = self.clone(tooling.CacheGrammar)
        return d

    @final
    def run(self):
        subject = subjects[self.format]
        automaton_dir = work_dir / "tribble-automaton-cache" / self.format
        grammar_file = Path("grammars") / subject["grammar"]
        tribble_jar = self.input()["tribble"].path
        # also make the seed depend on the output path starting from work_dir
        rel_output_path = Path(self.output().path).relative_to(work_dir)
        random_seed = self.random_int(self.tribble_seed, self.format, self.generation_mode, *rel_output_path.parts)
        with self.output().temporary_path() as out:
            args = ["java",
                    "-Xss100m",
                    "-Xms256m",
                    f"-Xmx{self.resources['ram']}g",
                    "-jar", tribble_jar,
                    f"--random-seed={random_seed}",
                    f"--automaton-dir={automaton_dir}",
                    "--no-check-duplicate-alts",
                    "generate",
                    f'--suffix={subject["suffix"]}',
                    f"--out-dir={out}",
                    f"--grammar-file={grammar_file}",
                    f"--mode={self.generation_mode}",
                    "--unfold-regexes",
                    "--merge-literals",
                    ]
            if config.use_grammar_caching:
                args.append(f"--grammar-cache-dir={self.input()['cache'].path}")
            else:
                args.append("--ignore-grammar-cache")
            logging.info("Launching %s", " ".join(args))
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL)


class WithGenerationMethodName(ABC):
    """Mixin providing the generation method name."""

    @property
    @abstractmethod
    def generation_method_name(self) -> str:
        raise NotImplementedError("You must name this generation method!")


class PredictorBasedInputGeneration(luigi.Task, utils.StableRandomness, utils.DynamicOutput, WithGenerationMethodName, metaclass=ABCMeta):
    """Generates inputs according to the given specs."""
    format: str = luigi.Parameter(description="The name of the format directory (e.g. json)", positional=False)
    alhazen_generation_seed: int = luigi.IntParameter(description="The seed for this generation run", positional=False, significant=False)

    @final
    def output(self) -> FileSystemTarget:
        return luigi.LocalTarget(work_dir / "inputs" / self.generation_method_name / self.format / self.subject_name)

    @final
    def run(self):
        with self.dynamic_output() as subject_out_dir:
            specs = list(Path(self.input().path).iterdir())
            yield [self.clone(GenerateInputsFromSpecFile,
                              format=self.format,
                              spec_file=str(spec_path),
                              output_dir=str(subject_out_dir / spec_path.stem),
                              seed=self.random_int(self.alhazen_generation_seed, self.generation_method_name, self.format, self.subject_name, spec_path.stem)
                              ) for spec_path in specs]


@requires(tooling.BuildAlhazen)
class GenerateInputsFromSpecFile(luigi.Task):
    """Generates a set of inputs from a given spec file."""
    format: str = luigi.Parameter(description="The name of the format directory (e.g. json)", positional=False)
    spec_file: str = luigi.Parameter(description="Path to the spec file", positional=False)
    output_dir: str = luigi.Parameter(description="Where to put the generated file", positional=False)
    seed: int = luigi.IntParameter(description="The seed for this generation run", positional=False, significant=False)
    resources = {"ram": 2}

    def output(self):
        return luigi.LocalTarget(self.output_dir)

    def run(self):
        subject = subjects[self.format]
        grammar_file = str(Path("grammars") / subject["grammar"])
        automaton_dir = str(work_dir / "tribble-automaton-cache" / self.format)
        args = ["java",
                "-Xss10m",
                "-Xms256m",
                f"-Xmx{self.resources['ram']}g",
                "-jar", self.input().path,
                "--grammar", grammar_file,
                "--automaton", automaton_dir,
                "--lengthMax", "30",
                "reachTargets",
                "--seed", str(self.seed),
                "--max-rule-occurrences", str(config.max_rule_occurrences),
                "--target", self.spec_file,
                "--suffix", subject["suffix"],
                "--output", self.output().path
                ]
        logging.info("Launching %s", " ".join(args))
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
