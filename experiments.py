#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import sys
from pathlib import Path
from pprint import pformat
from typing import Dict, List

import luigi
import nbconvert
import nbformat
import pandas as pd
from luigi.util import requires

from lib import config
from lib import counts_evaluation
from lib import gini_based
from lib import prediction_evaluation
from lib import random_based
from lib import subjects
from lib import support_based
from lib import tooling
from lib import tree_based
from lib import utils
from lib import work_dir


approaches = [tree_based.ComputePrecisionForWholeTree, random_based.ComputePrecisionForRandom]
if config.evaluate_strongest_support:
    approaches.append(support_based.ComputePrecisionForStrongestSupport)
if config.evaluate_lowest_gini:
    approaches.append(gini_based.ComputePrecisionForLowestGini)


@requires(*approaches)
class ComputeGenerationPrecision(luigi.Task):
    """Computes how many of the predictor-generated trees actually reach their targeted method for all generation methods."""

    def output(self):
        return luigi.LocalTarget(work_dir / "results" / "precision" / self.format / f"{self.subject_name}.csv")

    def run(self):
        result = pd.concat([pd.read_csv(x.path) for x in self.input()], sort=False)
        with open(self.output().path, "w", newline="") as out:
            result.to_csv(out, index=False)


class GenerateAndRunNotebook(luigi.Task, utils.StableRandomness):
    """Generates and executes a jupyter notebook detailing the experimental results."""
    random_seed: int = luigi.IntParameter(description="The main seed for this experiment. All other random seeds will be derived from this one.", positional=False)
    report_name: str = luigi.Parameter(description="The name of the report file.", positional=False, default="report")
    only_format: str = luigi.Parameter(description="Only run experiments for formats starting with this prefix.", positional=False, default="")

    @property
    def _selected_subjects(self) -> Dict[str, List[str]]:
        subject_info = {fmt: list(info["drivers"].keys()) for fmt, info in subjects.items() if fmt.startswith(self.only_format)}
        if not subject_info:
            raise ValueError(f"There are no formats starting with {self.only_format}!")
        return subject_info

    def requires(self):
        for fmt, drivers in self._selected_subjects.items():
            fmt_params = dict(format=fmt,
                              tribble_seed=self.random_int(self.random_seed, "tribble_seed", fmt),
                              training_seed=self.random_int(self.random_seed, "training_seed", fmt))
            yield tooling.ExtractFeatureTranslations(format=fmt)
            for subject in drivers:
                params = fmt_params.copy()
                params["subject_name"] = subject
                params["selection_seed"] = self.random_int(self.random_seed, "selection_seed", fmt, subject)
                params["subsampling_seed"] = self.random_int(self.random_seed, "subsampling_seed", fmt, subject)
                yield prediction_evaluation.EvaluateSubjectPredictionPower(**params)
                yield prediction_evaluation.EvaluateAveragePredictionPower(**params)
                params["alhazen_generation_seed"] = self.random_int(self.random_seed, "alhazen_generation_seed", fmt, subject)
                yield tree_based.CreateMethodNumberReport(**params)
                yield counts_evaluation.EvaluateConstrainedness(**params)
                params["random_generation_seed"] = self.random_int(self.random_seed, "random_generation_seed", fmt, subject)
                yield ComputeGenerationPrecision(**params)
                yield counts_evaluation.EvaluateReachedMethodCount(**params)
                if config.evaluate_target_method_execution_count:
                    yield counts_evaluation.EvaluateTargetMethodExecutionCount(**params)
                if config.evaluate_bycatch_execution_count:
                    yield counts_evaluation.EvaluateBycatchExecutionCount(**params)

    def output(self):
        return luigi.LocalTarget(work_dir / f"{self.report_name}.ipynb")

    def run(self):
        nb = nbformat.v4.new_notebook()
        cells = []
        cells.append(nbformat.v4.new_markdown_cell("# Imports"))
        cells.append(nbformat.v4.new_code_cell("""\
import plotly.express as px
import pandas as pd
pd.options.plotting.backend = "plotly"
pd.set_option("display.max_colwidth", None)"""))
        cells.append(nbformat.v4.new_markdown_cell("# Configuration"))
        cells.append(nbformat.v4.new_code_cell(f"""\
random_seed = {self.random_seed}
number_of_methods = {config.number_of_methods}
ignore_constructors = {repr(config.ignore_constructors)}
gini_threshold = {config.gini_threshold}
max_k = {config.max_k}
max_tree_depth = {config.max_tree_depth}
feature_variance_threshold = {config.feature_variance_threshold}
method_variance_threshold = {config.method_variance_threshold}
min_samples_leaf = {config.min_samples_leaf}
initial_inputs_generation_mode = {repr(config.initial_inputs_generation_mode)}
use_subsampling = {repr(config.use_subsampling)}
subsample_bias = {repr(config.subsample_bias)}
max_rule_occurrences = {config.max_rule_occurrences}
subjects = {pformat(self._selected_subjects, width=150)}"""))
        cells.append(nbformat.v4.new_markdown_cell("# Utility Functions"))
        cells.append(nbformat.v4.new_code_cell("""\
def load_reach_ratio(fmt: str, subject: str):
    return pd.read_csv(f"results/precision/{fmt}/{subject}.csv", index_col="target")

def concat_format(fmt: str):
    dfs = []
    for name in subjects[fmt]:
        df = load_reach_ratio(fmt, name)
        df["subject"] = name
        dfs.append(df)
    return pd.concat(dfs, sort=False)

def plot_generative_power(fmt: str):
    df = concat_format(fmt).drop(columns=["covered", "attempts"])
    fig = px.box(df, y="precision", x="subject", color="generation-method", points="all", title=f"Precision Distribution for {fmt} Subjects")
    fig.show()

def plot_prediction_power(fmt: str):
    dfs = [pd.read_csv(f"results/accuracy-evaluation/{fmt}/{subject}.csv") for subject in subjects[fmt]]
    dfs = pd.concat(dfs)
    melted = dfs.melt(id_vars=["subject","target"], var_name="characteristic")
    fig = px.box(melted, y="value", x="subject", color="characteristic", points="all", title=f"Prediction Power for {fmt} Subjects")
    fig.show()

def show_average_prediction_power(fmt: str):
    return pd.concat(pd.read_csv(f"results/prediction-evaluation/{fmt}/{subject}.csv", index_col=["format", "subject"]) for subject in subjects[fmt])

def show_average_covered_methods(fmt: str):
    return pd.concat(pd.read_csv(f"results/constrainedness/{fmt}/{subject}.csv", index_col=["format", "subject"]) for subject in subjects[fmt])

def show_method_numbers(fmt: str):
    return pd.concat(pd.read_csv(f"results/method-numbers/{fmt}/{subject}.csv", index_col=["format", "subject"]) for subject in subjects[fmt])

def _color_p_value(p):
    color = "green" if p <= 0.05 else "orange" if 0.05 < p <= 0.1 else "black"
    return f"color: {color}"

def _compare_stats(fmt: str, folder: str):
    df = pd.concat(pd.read_csv(f"results/{folder}/{fmt}/{subject}.csv", index_col=["format", "subject"]) for subject in subjects[fmt])
    return df.style.applymap(_color_p_value, subset=["p-value (two-sided)", "p-value (greater)"])

def compare_reached_methods(fmt: str):
    return _compare_stats(fmt, "method-counts")
"""))
        if config.evaluate_target_method_execution_count:
            cells.append(nbformat.v4.new_code_cell("""\
def compare_target_method_stats(fmt: str):
    return _compare_stats(fmt, "target-method-stats")
"""))
        if config.evaluate_bycatch_execution_count:
            cells.append(nbformat.v4.new_code_cell("""\
def compare_bycatch_stats(fmt: str):
    return _compare_stats(fmt, "bycatch-stats")

def plot_bycatch_counts(fmt: str):
    for name in subjects[fmt]:
        df = pd.read_csv(f"results/bycatch-counts/{fmt}/{name}.csv", index_col=["format", "subject", "targeted-method", "observed-method"])\\
            .sort_values(by="executions_random")\\
            .reset_index(drop=True)
        fig = df.plot.line(title=f"Bycatch on {name} ({fmt})")
        fig.show()
"""))
        cells.append(nbformat.v4.new_markdown_cell("# Evaluation"))
        for fmt in self._selected_subjects.keys():
            cells.append(nbformat.v4.new_markdown_cell(f"## Format {fmt}"))
            cells.append(nbformat.v4.new_code_cell(f"plot_generative_power({repr(fmt)})"))
            cells.append(nbformat.v4.new_markdown_cell(
                "The above plot shows the _generative power_ of the learned trees. " +
                "For each subject and approach it shows the ratio of the trees reaching the targeted method " +
                "averaged over all methods."))
            cells.append(nbformat.v4.new_code_cell(f"plot_prediction_power({repr(fmt)})"))
            cells.append(nbformat.v4.new_markdown_cell(
                "The above plot shows the _predictive power_ of the learned trees. " +
                "For each subject and method it shows in how many cases its tree predicted the correct outcome for whether an input will cover the method " +
                f"averaged over {config.initial_inputs_generation_mode.rsplit('-', 1)[-1]} inputs."))
            cells.append(nbformat.v4.new_code_cell(f"show_average_prediction_power({repr(fmt)})"))
            cells.append(nbformat.v4.new_markdown_cell("The above table shows the _predictive power_ of the trees averaged over all methods of each subject."))
            cells.append(nbformat.v4.new_code_cell(f"show_average_covered_methods({repr(fmt)})"))
            cells.append(nbformat.v4.new_markdown_cell("The above table shows the average number of methods covered by each approach for every subject."))
            cells.append(nbformat.v4.new_code_cell(f"compare_reached_methods({repr(fmt)})"))
            cells.append(nbformat.v4.new_markdown_cell("The above table compares the number of methods that were executed by the whole-tree inputs sets "
                                                       "with the number of methods that were executed by comparable randomly generated sets.  \n"
                                                       "The Wilcoxon signed-rank test shows whether these numbers differ significantly (`two-sided`) "
                                                       "and whether the whole-tree sets cover more methods compared to random sets (`greater`)."))
            cells.append(nbformat.v4.new_code_cell(f"show_method_numbers({repr(fmt)})"))
            cells.append(nbformat.v4.new_markdown_cell(f"The above table shows the number of methods{' (ignoring constructors) ' if config.ignore_constructors else ' '}"
                                                       "at every stage of the experimental pipeline:\n"
                                                       "- `observed` shows how many different methods were observed during the execution of the training input set\n"
                                                       "- `trainable` corresponds to the number of methods that were observed as both *covered* and *not covered*\n"
                                                       "- `selected` is the number of methods that were randomly selected for the experiment\n"
                                                       "- `generated` is the number of methods for which our approach was able to generate inputs\n"
                                                       "- `reached` shows how many methods we could reach with our generated inputs"))
            if config.evaluate_target_method_execution_count:
                cells.append(nbformat.v4.new_code_cell(f"compare_target_method_stats({repr(fmt)})"))
                cells.append(nbformat.v4.new_markdown_cell("The above table compares the number of times the targeted method was executed by the whole-tree inputs sets "
                                                           "with the number of times it was executed by comparable randomly generated sets.  \n"
                                                           "The Wilcoxon signed-rank test shows whether these execution numbers differ significantly (`two-sided`) "
                                                           "and whether the whole-tree sets have larger execution numbers compared to random sets (`greater`)."))
            if config.evaluate_bycatch_execution_count:
                cells.append(nbformat.v4.new_code_cell(f"compare_bycatch_stats({repr(fmt)})"))
                cells.append(nbformat.v4.new_markdown_cell("The above table compares the number of times non-targeted methods were executed by the whole-tree inputs sets "
                                                           "with the number of times they were executed by comparable randomly generated sets.  \n"
                                                           "The Wilcoxon signed-rank test shows whether these execution numbers differ significantly (`two-sided`) "
                                                           "and whether the whole-tree sets have larger execution numbers compared to random sets (`greater`)."))
                cells.append(nbformat.v4.new_code_cell(f"plot_bycatch_counts({repr(fmt)})"))

        nb["cells"] = cells

        ep = nbconvert.preprocessors.ExecutePreprocessor(kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": str(work_dir)}})

        with self.output().open("w") as f:
            nbformat.write(nb, f)


@requires(GenerateAndRunNotebook)
class RenderNotebook(luigi.Task):
    """Renders the experiment results as static html."""

    def output(self):
        return luigi.LocalTarget(Path(self.input().path).with_suffix(".html"))

    def run(self):
        html_exporter = nbconvert.HTMLExporter()
        html, _ = html_exporter.from_filename(self.input().path)
        with Path(self.output().path).open("w", encoding="utf-8") as f:
            f.write(html)


if __name__ == '__main__':
    luigi.BoolParameter.parsing = luigi.BoolParameter.EXPLICIT_PARSING
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", datefmt="%d.%m.%Y %H:%M:%S", level=logging.INFO, stream=sys.stdout)
    slack = luigi.configuration.get_config().get("luigi-monitor", "slack_url", None)
    if slack:
        import luigi_monitor

        with luigi_monitor.monitor(slack_url=slack):
            luigi.run(main_task_cls=RenderNotebook)
    else:
        luigi.run(main_task_cls=RenderNotebook)
