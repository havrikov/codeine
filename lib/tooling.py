#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains luigi tasks to build the tools required for the experiments.
"""

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import List

import luigi
from luigi.util import requires

from lib import config
from lib import subjects
from lib import tool_dir
from lib import work_dir


class GradleTask(object):
    @staticmethod
    def gradlew(*commands: str) -> List[str]:
        """Constructs a platform-appropriate gradle wrapper call string."""
        invocation = ["cmd", "/c", "gradlew.bat"] if platform.system() == "Windows" else ["./gradlew"]
        if commands:
            invocation.extend(commands)
        return invocation


class BuildMexCounter(luigi.Task, GradleTask):
    """Builds the mexcounter java agent and copies it into the working directory."""

    def output(self):
        return luigi.LocalTarget(work_dir / "tools" / "mexcounter.jar")

    def run(self):
        subprocess.run(self.gradlew("build"), check=True, cwd=tool_dir / "mexcounter", stdout=subprocess.DEVNULL)
        build_dir = tool_dir / "mexcounter" / "build" / "libs"
        artifact = next(build_dir.glob("**/mexcounter*.jar"))
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        shutil.copy(str(artifact), self.output().path)


class BuildAlhazen(luigi.Task):
    """Builds the alhazen jar and copies it into the working directory."""

    def output(self):
        return luigi.LocalTarget(tool_dir / "alhazen.jar")


class BuildTribble(luigi.Task, GradleTask):
    """Builds the tribble jar and copies it into the working directory."""

    def output(self):
        return luigi.LocalTarget(work_dir / "tools" / "tribble.jar")

    def run(self):
        subprocess.run(self.gradlew("assemble", "-p", "tribble-tool"), check=True, cwd=tool_dir / "tribble", stdout=subprocess.DEVNULL)
        build_dir = tool_dir / "tribble" / "tribble-tool" / "build" / "libs"
        artifact = next(build_dir.glob("**/tribble*.jar"))
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        shutil.copy(str(artifact), self.output().path)


@requires(BuildTribble)
class CacheGrammar(luigi.Task):
    """Generates a binary grammar cache for the given format."""
    format: str = luigi.Parameter(description="The name of the format directory (e.g. json)", positional=False)

    def output(self):
        return luigi.LocalTarget(work_dir / "tribble-grammar-cache" / self.format)

    def run(self):
        subject = subjects[self.format]
        automaton_dir = work_dir / "tribble-automaton-cache" / self.format
        grammar_file = Path("grammars") / subject["grammar"]
        tribble_jar = self.input().path
        with self.output().temporary_path() as out:
            args = ["java",
                    "-Xss100m",
                    "-Xms256m",
                    "-jar", tribble_jar,
                    f"--automaton-dir={automaton_dir}",
                    "--no-check-duplicate-alts",
                    "cache-grammar",
                    "--unfold-regexes",
                    "--merge-literals",
                    f"--grammar-cache-dir={out}",
                    f"--grammar-file={grammar_file}",
                    ]
            logging.info("Launching %s", " ".join(args))
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL)


class BuildSubject(luigi.Task, GradleTask):
    """Builds the given subject and copies it into the working directory."""
    subject_name: str = luigi.Parameter(description="The name of the subject to build", positional=False)

    def output(self):
        return luigi.LocalTarget(work_dir / "tools" / "subjects" / self.subject_name / f"{self.subject_name}-subject.jar")

    def run(self):
        subprocess.run(self.gradlew("build", "-p", self.subject_name), check=True, cwd=tool_dir / "subjects", stdout=subprocess.DEVNULL)
        artifact = tool_dir / "subjects" / self.subject_name / "build" / "libs" / f"{self.subject_name}-subject.jar"
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        shutil.copy(str(artifact), self.output().path)


class DownloadOriginalBytecode(luigi.Task, GradleTask):
    """Downloads the unmodified bytecode of the subject and places it into the working directory."""
    subject_name: str = luigi.Parameter(description="The name of the subject to build")

    def output(self):
        return luigi.LocalTarget(work_dir / "tools" / "subjects" / self.subject_name / f"{self.subject_name}-original.jar")

    def run(self):
        subprocess.run(self.gradlew("downloadOriginalJar", "-p", self.subject_name), check=True, cwd=tool_dir / "subjects", stdout=subprocess.DEVNULL)
        artifact = tool_dir / "subjects" / self.subject_name / "build" / "libs" / f"{self.subject_name}-original.jar"
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        shutil.copy(str(artifact), self.output().path)


@requires(BuildAlhazen)
class ExtractFeatureTranslations(luigi.Task):
    """Creates a csv file which maps feature names to a readable representation."""
    format: str = luigi.Parameter(description="The name of the format directory (e.g. json)", positional=False)
    resources = {"ram": 1}

    def output(self):
        return luigi.LocalTarget(work_dir / "patterns" / f"{self.format}.csv")

    def run(self):
        grammar_file = str(Path("grammars") / subjects[self.format]["grammar"])
        automaton_dir = str(work_dir / "tribble-automaton-cache" / self.format)
        tmp_grammar_output = work_dir / "grammars"
        tmp_grammar_output.mkdir(parents=True, exist_ok=True)
        with self.output().temporary_path() as res:
            args = ["java",
                    "-Xss10m",
                    "-Xms256m",
                    f"-Xmx{self.resources['ram']}g",
                    "-jar", self.input().path,
                    "--grammar", grammar_file,
                    "--automaton", automaton_dir,
                    "rule-depth",
                    "--present-absent",
                    "--k-paths", str(config.max_k),
                    "--output", res,
                    "--grammar-output", str(tmp_grammar_output / f"{self.format}.rewritten_grammar.scala")
                    ]
            logging.info("Launching %s", " ".join(args))
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
