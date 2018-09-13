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

    def __init__(self, dims, size, autoprune=True):

        self._size = size
        self._autoprune = autoprune
        self._dimensions = dims
        Node._init_branch_matrix(self._dimensions)

        self._branch_factor = self._dimensions * 2
        root = Node(None, None, dims)
        self._nodes = [root]
        self._root = root

        self._min_level = 0

        init_level = compute_level(size, self._branch_factor)
        for i in range(init_level):
            self.add_layer()

        self._total_distance = 0
        self._total_distance_count = 0

    def search_nearest_node(self, point, increase=True):

        point = self.correct_point(point.flatten())

        node, dist = self._root.search(point, increase)

        self._total_distance += dist

        self._total_distance_count += 1

        return node

    def update(self):
        _points_before = np.array(self.get_points())

        to_prune = self.prune_prospectives()
        pruned = 0
        for node in to_prune:
            node.delete()
            pruned += 1

        excess = self.get_current_size() - self.get_size()
        expanded = self.expand_usefull_nodes(pruned - excess)

        self._refresh_nodes()
        self._reset_values()

        _points_after = np.array(self.get_points())

        if pruned == expanded:
            for i in range(len(_points_after)):
                if np.linalg.norm(_points_before[i] - _points_after[i]) > 0:
                    return True
            return False
        else:
            return True

    def feed(self, samples):
        for sample in samples:
            self.search_nearest_node(sample)

    def adapt_to_1D_pdf(self, pdf, sample_to_resolution_ratio=10, max_iterations=10):
        resolution = len(pdf)
        x = np.linspace(0, 1, resolution, endpoint=True)
        samples = np.random.choice(x, resolution*sample_to_resolution_ratio, p=pdf.flatten())

        print("Adaption begun,", len(samples), "samples, max iterations", max_iterations)
        count = 0
        flag = True
        while flag and count <= max_iterations:
            self.feed(samples)
            flag = self.update()
            print("Iteration", count, ", adapted:", not flag)
            count += 1

    def prune_prospectives(self):

        nodes = self.get_prunable_nodes()

        mean_value = self.get_mean_value()

        result = []

        for node in nodes:
            estimated_future_value = node.get_value() + node.get_value_increase_if_cut()
            if(estimated_future_value < mean_value):
                result.append(node)

        return result

    def expand_usefull_nodes(self, n):
        nodes = sorted(self.get_nodes(recalculate=True), key=lambda node: node.get_value())
        suggestions = list(node.suggest_for_expand() for node in nodes)

        count_expantions = 0
        i = len(nodes)
        while count_expantions < n and i >= 0:
            i -= 1
            if(nodes[i].get_value() == 0):
                continue

            to_expand = suggestions[i]
            new_nodes = []
            for suggestion in to_expand:
                max_new_nodes = n - count_expantions - len(new_nodes)
                new_nodes.extend(suggestion.expand(
                    nodes[i].get_location(), new_nodes_limit=max_new_nodes))

            self._nodes.extend(new_nodes)
            count_expantions += len(new_nodes)

        return count_expantions

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

    def get_mean_error(self):
        if self._total_distance_count == 0:
            return 0

        return np.sum(self.get_values()) / self._total_distance_count

    def get_mean_value(self):
        return np.sum(self.get_values()) / self.get_current_size()

    def _get_max_mean_distance(self):
        return 1 / (4 * self.get_current_size())

    def _reset_values(self):
        self.recursive_traversal(func=lambda node: node.reset_value())
        self._total_distance = 0
        self._total_distance_count = 0

    def add_layer(self):
        to_expand = self.get_expendable_nodes()
        for node in to_expand:
            new_nodes = node.expand()
            self._nodes.extend(new_nodes)

    def get_nodes(self, recalculate=False):
        if recalculate:
            self._nodes = self.recursive_traversal()
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

    def plot(self, red_levels=False, save=False, filename=''):
        import tree_vis
        tree_vis.plot(self, save=save, red_levels=red_levels, path=filename)

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
