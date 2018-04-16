#!/usr/bin/python3
import numpy as np
import math

from node import *


def compute_level(n, branches_of_each_node):
    total = 0
    power = -1
    prev = 0
    while total <= n:
        power += 1
        prev = total
        total += branches_of_each_node**power

    if total - n > n - prev:
        return max(power - 1, 0)
    return max(power, 0)


class Tree:

    def __init__(self, dims, avg_nodes, autoprune=True):

        self._size = avg_nodes
        self._autoprune = autoprune
        self._dimensions = dims
        Node._init_branch_matrix(self._dimensions)

        self._branch_factor = self._dimensions * 2
        root = Node(np.ones(dims) * 0.5, None)
        self._nodes = [root]
        self._root = root

        self._min_level = 0

        init_level = compute_level(avg_nodes, self._branch_factor)
        for i in range(init_level):
            self.expand_nodes(0)

        self._total_distance = 0
        self._total_distance_count = 0

    def search_nearest_node(self, point, increase=True):

        point = self.correct_point(point)

        node, dist = self._root.search(point, increase)

        self._total_distance += dist

        self._total_distance_count += 1

        return node

    def update(self, reward_factor=1):
        # find the expand value
        selected_exp_value = self._expand_threshold_value(reward_factor)
        size_before = self.get_current_size()

        # expand nodes with values greater or equal to the choosen one
        self.expand_nodes(selected_exp_value)
        size_after = self.get_current_size()

        # find the cut value
        selected_cut_value = self._prune_threshold_value(max_threshold=selected_exp_value)
        assert selected_cut_value < selected_exp_value, 'cut value > expand value'

        prunable = self.get_prunable_nodes()
        # cut the nodes with values below (not equal) to the choosen one
        for node in prunable:
            if node.get_value() <= selected_cut_value:
                node.delete()

        # self.plot()

        self._refresh_nodes()
        self._reset_values()

    def _prune_threshold_value(self, max_threshold=np.inf):

        excess = self.get_current_size() - self.get_size()
        if excess <= 0:
            return -1

        values = list(node.get_value() for node in self.get_prunable_nodes())

        unique, counts = np.unique(values, return_counts=True)

        valid_values_indexex = np.where(unique < max_threshold)[0]

        unique = unique[valid_values_indexex]
        counts = counts[valid_values_indexex]

        unique = np.insert(unique, 0, -1)
        counts = np.insert(counts, 0, 0)

        for i in range(1, len(counts)):
            counts[i] += counts[i - 1]

        delta_size = np.abs(counts - excess)

        result_value = unique[np.argmin(delta_size)]

        return result_value

    def _expand_threshold_value(self, factor=1):

        factor = max(factor, .01)

        v_exp = list(node.get_value() for node in self.get_expendable_nodes())

        avg_v = np.average(v_exp)

        result_value = avg_v / factor

        return result_value

    def _evaluate(self):
        mean_distance = self._get_mean_distance()
        max_mean_distance = self._get_max_mean_distance()
        distance_factor = mean_distance / max_mean_distance
        return distance_factor

    def get_node(self, index):
        node = self.get_nodes()[index]
        return node

    def recursive_traversal(self, func=(lambda node: node),
                            traverse_cond_func=(lambda node: True),
                            collect_cond_func=(lambda node: True)):
        res = []
        self._root.recursive_collection(res, func, traverse_cond_func, collect_cond_func)
        return res

    def get_values(self):
        return self.recursive_traversal(func=(lambda node: node.get_value()))

    def get_total_value(self):
        return np.sum(self.get_values())

    def get_prunable_nodes(self):
        return self.recursive_traversal(collect_cond_func=(lambda node: node.is_leaf()))

    def get_expendable_nodes(self):
        return self.recursive_traversal(collect_cond_func=(lambda node: node.is_expandable()))

    def _get_mean_distance(self):
        if self._total_distance_count == 0:
            return 0
        # result = self._total_distance / self._total_distance_count

        return np.sum(self.get_values()) / self._total_distance_count

    def _get_max_mean_distance(self):
        return 1 / (4 * self.get_current_size())

    def _reset_values(self):
        self.recursive_traversal(func=lambda node: node.reset_value())
        self._total_distance = 0
        self._total_distance_count = 0

    def expand_nodes(self, value_threshold):
        to_expand = self.get_expendable_nodes()
        for node in to_expand:
            if node.get_value() >= value_threshold:
                new_nodes = node.expand()
                self._nodes.extend(new_nodes)

    def get_nodes(self):
        return self._nodes

    def _refresh_nodes(self):
        self._nodes = self.recursive_traversal()

    def get_points(self):
        return np.array(list(node.get_location() for node in self.get_nodes()))

    def get_current_size(self):
        return len(self._nodes)

    def get_size(self):
        return self._size

    def print_all_nodes(self):
        nodes = self._nodes
        print('tree contains', len(nodes), 'nodes, min level=', self._min_level)
        for node in nodes:
            print(node)

    def plot(self):
        import tree_vis
        tree_vis.plot(self)

    @staticmethod
    def correct_point(point):
        new_point = []
        for c in point:
            if c > 1:
                new_point.append(1)
            elif c < 0:
                new_point.append(0)
            else:
                new_point.append(c)

        return new_point
