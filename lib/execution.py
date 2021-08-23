#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains luigi tasks for executing the test subjects.
"""

import logging
import subprocess
from abc import ABCMeta
from abc import abstractmethod
from pathlib import Path
from typing import final

import luigi
from luigi.target import FileSystemTarget
from luigi.util import inherits

from lib import config
from lib import drivers
from lib import subjects
from lib import tooling
from lib import utils
from lib import work_dir


@inherits(tooling.BuildSubject, tooling.DownloadOriginalBytecode, tooling.BuildMexCounter)
class RunSubject(luigi.Task, utils.CoverageOutput, metaclass=ABCMeta):
    """Runs the given subject on the inputs produced by the given task and produces a method coverage report."""
    resources = {"ram": 1}

    @property
    @abstractmethod
    def input_task(self) -> luigi.Task:
        raise NotImplementedError("You must provide the task to produce the inputs to execute the subject with!")

    @abstractmethod
    def output(self) -> FileSystemTarget:
        raise NotImplementedError("You must point to where to put the csv.gz compressed execution report!")

    @final
    def requires(self):
        reqs = {
            "subject_jar": self.clone(tooling.BuildSubject),
            "original_jar": self.clone(tooling.DownloadOriginalBytecode),
            "inputs": self.input_task,
        }
        if config.needs_method_counter_instrumentation:
            reqs["mexcounter_jar"] = self.clone(tooling.BuildMexCounter)
        return reqs

    @final
    def run(self):
        subject_jar = self.input()["subject_jar"].path
        original_jar = self.input()["original_jar"].path
        input_path = self.input()["inputs"].path
        subject_package = drivers[self.subject_name]
        original_output = Path(self.output().path)
        args = ["java",
                "-Xss10m",
                "-Xms256m",
                f"-Xmx{self.resources['ram']}g",
                ]
        if config.needs_method_counter_instrumentation:
            mexcounter_agent_path = self.input()['mexcounter_jar'].path
            args.append(f"-javaagent:{mexcounter_agent_path}={subject_package},{self.hitcount_report(original_output)}")
        args.extend([
            "-jar", subject_jar,
            "--ignore-exceptions",
            "--report-methods", str(self.coverage_report(original_output)),
            "--original-bytecode", original_jar,
            input_path,
        ])
        logging.info("Launching %s", " ".join(args))
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL)

    @final
    def complete(self):
        original_output = Path(self.output().path)
        done = self.coverage_report(original_output).exists()
        if config.needs_method_counter_instrumentation:
            done = done and self.hitcount_report(original_output).exists()
        return done


@final
class RunSubjectOnDirectory(RunSubject):
    """Runs the given subject with the given inputs and produces a method coverage report at the given location."""
    input_directory: str = luigi.Parameter(description="The directory with inputs to feed into the subject.", positional=False)
    report_file: str = luigi.Parameter(description="The path to the .csv.gz compressed execution report.", positional=False)

    @property
    def input_task(self) -> luigi.Task:
        class OutputWrapper(luigi.Task):
            # This parameter is significant for luigi to distinguish between different directories
            input_dir: str = luigi.Parameter(description="Pretend that this task has output this dir")

            def output(self) -> FileSystemTarget:
                return luigi.LocalTarget(self.input_dir)

        return OutputWrapper(self.input_directory)

    def output(self) -> FileSystemTarget:
        return luigi.LocalTarget(self.report_file)


@inherits(tooling.BuildAlhazen)
class ExtractInputFeatures(luigi.Task, metaclass=ABCMeta):
    """Parses the inputs and extracts input features."""
    resources = {"ram": 1}

    @property
    @abstractmethod
    def input_task(self) -> luigi.Task:
        raise NotImplementedError("You must provide the task to generate the inputs!")

    @abstractmethod
    def output(self) -> FileSystemTarget:
        raise NotImplementedError("You must point to where to put the feature csv report!")

    @final
    def requires(self):
        return {
            "alhazen": self.clone(tooling.BuildAlhazen),
            "generated-files": self.input_task,
        }

    @final
    def run(self):
        subject = subjects[self.format]
        alhazen_jar = self.input()["alhazen"].path
        input_path = self.input()["generated-files"].path
        grammar_file = str(Path("grammars") / subject["grammar"])
        automaton_dir = str(work_dir / "tribble-automaton-cache" / self.format)
        with self.output().temporary_path() as feature_file:
            args = ["java",
                    "-Xss10m",
                    "-Xms256m",
                    f"-Xmx{self.resources['ram']}g",
                    "-jar", alhazen_jar,
                    "--grammar", grammar_file,
                    "--automaton", automaton_dir,
                    "countAmbiguous",
                    "--present-absent",
                    "--k-paths", str(config.max_k),
                    "--input", input_path,
                    "--output", feature_file,
                    ]
            logging.info("Launching %s", " ".join(args))
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
