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
        root = Node(None, None, dims)
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

        to_prune = self.prune_prospectives()
        pruned = 0
        for node in to_prune:
            # print(node)
            node.delete()
            pruned += 1
        # print('pruned', pruned)
        # self.plot()

        excess = self.get_current_size() - self.get_size()
        expanded = self.expand_usefull_nodes(pruned - excess)

        self._refresh_nodes()
        self._reset_values()

        return pruned == 0 and expanded == 0

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
        # print('able to expand', n)
        nodes = sorted(self.get_nodes(recalculate=True), key=lambda node: node.get_value())
        suggestions = list(node.suggest_for_expand() for node in nodes)

        count_expantions = 0
        i = len(nodes)
        while count_expantions < n and i >= 0:
            i -= 1
            if(nodes[i].get_value() == 0):
                continue

            to_expand = suggestions[i]
            # print(to_expand)
            new_nodes = []
            for suggestion in to_expand:
                new_nodes.extend(suggestion.expand(nodes[i].get_location()))

            # print(nodes[i], 'suggests', to_expand, 'and expanded to ', new_nodes)
            # print(nodes[i], len(new_nodes), new_nodes)
            self._nodes.extend(new_nodes)
            # print(len(new_nodes), new_nodes)
            count_expantions += len(new_nodes)

        # print(count_expantions, 'expansions, i=', i)
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
        # result = self._total_distance / self._total_distance_count

        return np.sum(self.get_values()) / self._total_distance_count

    def get_mean_value(self):
        return np.sum(self.get_values()) / self.get_current_size()

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
