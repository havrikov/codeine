#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Optional, Dict, Iterator

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class TreeHelper(object):
    """Defines helper methods for interacting with a decision tree."""

    def __init__(self, clf: DecisionTreeClassifier, features: List[str]):
        self.clf = clf
        self.features = features

    def strongest_support_path(self, gini_threshold: float = 1.0) -> Dict[str, bool]:
        """Returns the covering path with the greatest support (shortest if multiple).

        gini_threshold: float - only consider paths ending in leaves with a gini impurity <= threshold
        """
        covering_paths = self._covering_paths(gini_threshold)
        best_path = max(covering_paths, key=lambda p: self.clf.tree_.n_node_samples[p[-1]], default=[])
        return self.feature_mapping(best_path)

    def strongest_gini_path(self) -> Dict[str, bool]:
        """Returns the covering path with the lowest gini impurity (shortest if multiple)."""
        covering_paths = self._covering_paths(1.0)
        best_path = min(covering_paths, key=lambda p: self.clf.tree_.impurity[p[-1]], default=[])
        return self.feature_mapping(best_path)

    def _covering_paths(self, gini_threshold: float) -> Iterator[List[int]]:
        """Returns all covering paths in ascending order by length.

        gini_threshold: float - only return paths ending in leaves with a gini impurity <= threshold
        """
        covering_paths = filter(self._is_path_covering, self.all_paths())
        matching_paths = (p for p in covering_paths if self.clf.tree_.impurity[p[-1]] <= gini_threshold)
        length_ascending = sorted(matching_paths, key=len)
        return length_ascending

    def covering_specifications(self, gini_threshold: float = 1.0) -> Iterator[Dict[str, bool]]:
        """Returns a mapping of features to whether they are required or prohibited for each path in the tree that ends in a leaf of class 'covered'.

        gini_threshold: float - only return paths ending in leaves with a gini impurity <= threshold
        """
        return map(self.feature_mapping, self._covering_paths(gini_threshold))

    def all_paths(self, node: int = 0) -> List[List[int]]:
        """
        Iterate over all paths in a decision tree.
        A path is represented as a list of integers, each integer is the index of a node in the clf.tree_ structure.
        """
        left = self.clf.tree_.children_left[node]
        right = self.clf.tree_.children_right[node]
        if left == right:
            yield [node]
        else:
            for path in self.all_paths(left):
                yield [node] + path
            for path in self.all_paths(right):
                yield [node] + path

    def feature_mapping(self, path: List[int]) -> Dict[str, bool]:
        """Returns a mapping of features to whether they are required or prohibited for the given path."""
        mapping = {}
        for pos in range(len(path) - 1):
            node = path[pos]
            child = path[pos + 1]
            feature = self._feature_for_node(node)
            req = self._presence_required(child, node)
            mapping[feature] = req
        return mapping

    def _is_path_covering(self, path: List[int]) -> bool:
        """Tests if a path leads to classifying an input as covering."""
        last_value = self.clf.tree_.value[path[-1]][0]
        p_class = np.argmax(last_value)
        return self.clf.classes_[p_class]

    def _feature_for_node(self, node: int) -> Optional[str]:
        """Returns the feature name for the given node."""
        idx = self.clf.tree_.feature[node]
        return None if idx < 0 else self.features[idx]

    def _presence_required(self, child_node: int, parent_node: int) -> bool:
        """Tests whether the feature described by the child_node is required given its parent node (otherwise it is prohibited)."""
        return child_node == self.clf.tree_.children_right[parent_node]
