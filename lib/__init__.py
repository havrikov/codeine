from collections import ChainMap
from pathlib import Path
from typing import Any, Dict

import luigi

from lib import subject_collection


class ExperimentConfig(luigi.Config):
    experiment_dir: str = luigi.Parameter(description="The path to where all the experiments happen. Should be outside the repository.")
    tool_dir: str = luigi.Parameter(description="The path to the tool sources.")
    gini_threshold: float = luigi.FloatParameter(description="Only consider paths ending in leaves with a gini impurity <= threshold.")
    max_k: int = luigi.IntParameter(description="The maximum k to consider for k-path features.")
    number_of_methods: int = luigi.IntParameter(description="How many methods to select.")
    ignore_constructors: bool = luigi.BoolParameter(description="Do not count constructor invocations as methods.")
    use_grammar_caching: bool = luigi.BoolParameter(description="Let tribble cache grammar files so it does not have to re-parse them on each invocation.")
    initial_inputs_generation_mode: str = luigi.Parameter(description="Which mode to use when generating the initial inputs to learn from.")
    generation_mode_prefix: str = luigi.Parameter(description="The prefix of the mode to use for generating random files to compare against those generated with the learned trees.")
    remove_randomly_generated_files: bool = luigi.BoolParameter(description="Remove the randomly generated files after we have acquired the execution metrics to save space. "
                                                                            "This does not apply to the guided approaches such as whole-tree, gini-based, or support-based")
    feature_variance_threshold: float = luigi.FloatParameter(description="The fraction of inputs the features may have the same value. Must be in [0,0.5].")
    method_variance_threshold: float = luigi.FloatParameter(description="The fraction of inputs the method coverage may have the same value. Must be in [0,0.5].")
    use_subsampling: bool = luigi.BoolParameter(description="Use subsampling instead of a variance threshold to filter out invariant methods.")
    subsample_bias: bool = luigi.BoolParameter(description="Bias subsampling so that only the non-covered class can be reduced.")
    max_tree_depth: int = luigi.IntParameter(description="The maximum depth of the trained decision trees.")
    max_rule_occurrences: int = luigi.IntParameter(description="The maximum number of the same declarations on the path from the root to a leaf.")
    min_samples_leaf: float = luigi.FloatParameter(description="The minimum fraction of samples that are required to form a decision tree leaf.")
    evaluate_strongest_support: bool = luigi.BoolParameter(description="Additionally evaluate the generative power of the tree paths with the strongest support.")
    evaluate_lowest_gini: bool = luigi.BoolParameter(description="Additionally evaluate the generative power of the tree paths with the lowest gini.")
    evaluate_target_method_execution_count: bool = luigi.BoolParameter(description="Enable evaluating how often the targeted method is executed between whole-tree and random approaches.")
    evaluate_bycatch_execution_count: bool = luigi.BoolParameter(description="Add an evaluation of how often methods other than the targeted method are executed between whole-tree and random approaches.")

    @property
    def needs_method_counter_instrumentation(self) -> bool:
        return self.evaluate_bycatch_execution_count or self.evaluate_target_method_execution_count


config = ExperimentConfig()
work_dir: Path = Path(config.experiment_dir)
tool_dir: Path = Path(config.tool_dir)

subjects: Dict[str, Dict[str, Any]] = subject_collection.Subjects().all_subjects

drivers = ChainMap(*[d["drivers"] for d in subjects.values()])
