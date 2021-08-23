#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains various utility mixins.
"""
import hashlib
import random
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Union, Final

import math
import pandas as pd
from luigi.target import FileSystemTarget
from scipy.stats import wilcoxon

from lib import config


class MethodCSVUtils(object):
    """This mixin provides a method to read a csv as produced by the instrumented subject and produce a coverage report."""

    @staticmethod
    def read_coverage_report(path: Union[Path, str]) -> pd.DataFrame:
        # the gzipped csv is assumed to be structured like this: "input_file", "class_name", "method_name", "line", "instructions_missed", "instructions_hit"
        methods = pd.read_csv(path, usecols=["class_name", "method_name", "input_file", "instructions_hit"])
        # create the column which will have 1 if a method is covered and 0 otherwise
        methods["covered"] = (methods.instructions_hit > 0).astype(int)
        # create a column with qualified method names in the form of ClassName::methodName
        methods["target"] = methods["class_name"].str.cat(methods.method_name, sep="::")
        # filter out constructors and static class initializers if requested
        if config.ignore_constructors:
            methods = methods[~methods["target"].str.contains("<(?:cl)?init>")]
        # drop unnecessary columns
        methods.drop(columns=["instructions_hit", "class_name", "method_name"], inplace=True)
        # count a polymorphic method as covered if any of its variants is covered
        return methods.groupby([methods.input_file, methods.target], as_index=False).aggregate({"covered": "max"})


class StableRandomness(object):
    """This mixin provides a method for generating random ints from a seed and a list of strings."""
    # For use with random.randrange()
    # Not using sys.maxsize because it might differ depending on the environment
    MAX_RND_INT: Final[int] = 2 ** 32 - 1

    @staticmethod
    def get_random(seed: int, *args: str) -> random.Random:
        """Get a random.Random instance initialized with a seed derived from the the given args."""
        # compute a stable hashcode of arguments as a string
        concat = ''.join(args)
        hash_code = int(hashlib.sha1(concat.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
        return random.Random(seed + hash_code)

    @staticmethod
    def random_int(seed: int, *args: str) -> int:
        """Get a random int that is uniquely derived from the given args."""
        rnd = StableRandomness.get_random(seed, *args)
        return rnd.randrange(StableRandomness.MAX_RND_INT)


class DynamicOutput(ABC):
    """This mixin adds the dynamic_output() method to use with dynamically spawned tasks."""

    @contextmanager
    def dynamic_output(self) -> Path:
        """Decorator creating a temporary .inprogress directory for dynamically spawned tasks to write into and atomically rename it into the original output path once they all succeed."""
        original: FileSystemTarget = self.output()
        tmp = Path(original.path).with_suffix(".inprogress")
        yield tmp
        # unreachable in case of an exception
        original.fs.rename_dont_move(tmp, original.path)

    @abstractmethod
    def output(self) -> FileSystemTarget:
        raise NotImplementedError("You must provide an output directory!")


def escape_method_name(s: str) -> str:
    """Escapes class and method names to be compatible with the file system."""
    return s.translate(str.maketrans({k: "_" for k in "$:<>/."}))


class CoverageOutput(object):
    """This mixin helps deal with "coverage" and "hitcounts" subdirectories in task outputs."""
    coverage_dir_name: Final[str] = "coverage"
    hitcounts_dir_name: Final[str] = "hitcounts"

    def coverage_report(self, path: Union[Path, str]) -> Path:
        """Returns the jacoco coverage report path."""
        return self._subdir(path, self.coverage_dir_name)

    def hitcount_report(self, path: Union[Path, str]) -> Path:
        """Returns the method hitcount report file."""
        p = Path(path)
        return self._subdir(p, self.hitcounts_dir_name).with_name(p.stem.split(".")[0]).with_suffix(".hitcount.csv")

    def read_hitcounts(self, path: Union[str, Path]) -> pd.DataFrame:
        """Reads the given .hitcounts.csv file and returns a data frame of the form method,executions."""
        df = pd.read_csv(self.hitcount_report(path))
        # strip out the signature and escape special characters
        df["method"] = df["method"].str.split("(").str[0].apply(escape_method_name)
        # just as with the coverage, count all polymorphic variants as one
        return df.groupby("method").aggregate({"executions": "max"})

    @staticmethod
    def _subdir(path: Union[Path, str], subdir: str) -> Path:
        """Inserts subdir at the second to last position in the given path."""
        p = Path(path)
        return p.parent.joinpath(subdir, p.name)


def remove_tree(path: Path) -> None:
    """Recursively removes the entire file tree under and including the given path."""
    if path.is_dir():
        for child in path.iterdir():
            remove_tree(child)
        path.rmdir()
    else:
        path.unlink()


def safe_wilcoxon(diffs, **kwargs):
    """Return NaN if all diffs are zero. Otherwise carry out the wilcoxon test."""
    return wilcoxon(diffs, **kwargs) if any(diffs) else (math.nan, math.nan)


def _percentage_to_variance(fraction: float) -> float:
    """
    Converts expected fraction of variant boolean values into a variance threshold
    as per https://sklearn.org/modules/feature_selection.html#removing-features-with-low-variance
    """
    if not (0.0 <= fraction <= 0.5):
        raise ValueError(f"Variance fraction must be in [0,0.5]. (Was {fraction})")
    return fraction * (1.0 - fraction)
