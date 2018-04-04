#!/usr/bin/python3
import numpy as np
import math

import matplotlib.pyplot as plt

from node import *


class Exploration_tree:

    EXPANSION_VALUE_THRESHOLD = 1

    def __init__(self, dims, avg_nodes, autoprune=True):

        self._size = avg_nodes
        self._autoprune = autoprune
        self._dimensions = dims
        Node._init_branch_matrix(self._dimensions)

        self._branch_factor = self._dimensions * 2
        root = Node(np.ones(dims) * 0.5, 0.5, None)
        self._nodes = [root]
        self._root = root

        self._min_level = 0

        init_level = Exploration_tree.compute_level(avg_nodes, self._branch_factor)
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
        # print('------------------UPDATE---------------')
        debug_flag = False

        if debug_flag:
            nodes = list(({'loc': node.get_location()[0], 'v': node.get_value()}
                          for node in self.get_expendable_nodes()))
            nodes = sorted(nodes, key=lambda node: node['loc'])
            points = list(item['loc'] for item in nodes)
            values = list(item['v'] for item in nodes)
            max_v = np.max(values)
            plt.plot(points, values, 'b^--', label='values (max={})'.format(max_v))

        # find the expand value
        selected_exp_value = self._expand_threshold_value(reward_factor)
        size_before = self.get_current_size()
        # expand nodes with values greater or equal to the choosen one
        self.expand_nodes(selected_exp_value)
        size_after = self.get_current_size()
        # print('selected values exp index, value', selected_exp_value,
        #       '# new nodes', size_after - size_before)

        # find the cut value
        selected_cut_value = self._prune_threshold_value(max_threshold=selected_exp_value)
        assert selected_cut_value < selected_exp_value, 'cut value > expand value'

        size_before = self.get_current_size()
        to_cut = self.get_prunable_nodes()
        # cut the nodes with values below (not equal) to the choosen one
        for node in to_cut:
            if node.get_value() <= selected_cut_value:
                node.delete()

        # self.plot()

        self._refresh_nodes()

        ######
        if debug_flag:
            nodes = list(({'loc': node.get_location()[0], 'v': node.get_value()}
                          for node in self.get_expendable_nodes()))
            nodes = sorted(nodes, key=lambda node: node['loc'])
            points = list(item['loc'] for item in nodes)
            values = list(item['v'] for item in nodes)
            max_v = np.max(values)
            # values = values / max_v
            # values = apply_func_to_window(values, int(.1 * len(values)), np.average)

            plt.plot([0, 1], [selected_exp_value, selected_exp_value],
                     'g', label='exp = {}'.format(selected_exp_value))

            plt.plot([0, 1], [selected_cut_value, selected_cut_value],
                     'r', label='cut = {}'.format(selected_cut_value))
            plt.plot(points, values, 'mv--', label='values (max={})'.format(max_v))
            plt.grid(True)
            plt.legend()
            plt.show()
        ####

        self._reset_values()

        # size_after = self.get_current_size()
        # print('selected values cut index, value', selected_cut_value,
        #       '# deleted nodes', size_after - size_before)

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

    def get_prunable_nodes(self):
        return self.recursive_traversal(collect_cond_func=(lambda node: node.is_leaf()))

    def get_expendable_nodes(self):
        return self.recursive_traversal(collect_cond_func=(lambda node: node.is_expandable()))

    def _get_mean_distance(self):
        if self._total_distance_count == 0:
            return 0
        result = self._total_distance / self._total_distance_count

        return result

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

    SAVE_ID = 0

    def plot(self, save=False, path='/home/jim/Desktop/dip/Adaptation-of-Action-Space-for-Reinforcement-Learning/results/pics'):
        nodes = self.get_nodes()
        plt.figure()
        print('nodes to plot:', len(nodes))
        max_level = np.max(list(node.get_level() for node in nodes))
        plt.title('tree size={}'.format(len(nodes)))
        for node in nodes:
            parent, child = node.get_connection_with_parent()
            r = 0
            b = 0
            # if node.get_value() == 0 and node._parent is not None:

            # if node.is_expandable():
            #     b = 255
            # if node.is_leaf():
            #     r = 255

            if self._dimensions == 1:
                if node.is_root():
                    x = [child[0]]
                    y = [node._level]
                else:
                    x = [child[0], parent[0]]
                    y = [node._level, node._level - 1]

                plt.yticks(range(max_level + 1))

                plt.plot([x, x], [-0.1, -0.2], '#000000', linewidth=0.5)

            else:
                x = [child[0], parent[0]]
                y = [child[1], parent[1]]

                plt.plot([child[0], child[0]], [-0.1, -0.12], '#000000', linewidth=.5)
                plt.plot([-0.1, -0.12], [child[1], child[1]], '#000000', linewidth=.5)

            plt.plot(x, y, 'm-', linewidth=0.2)
            size = 2.1 * (1 + max_level - node.get_level())
            plt.plot(x[0], y[0],
                     '#{:02x}00{:02x}'.format(r, b), marker='.', markersize=size)

        # if self._dimensions == 1:
        #     # f = 0.1
        #     # hist, x_h = np.histogram(self.get_points().flatten(), bins=int(len(nodes) * f))
        #     # x_h = list((x_h[i] + x_h[i - 1]) / 2 for i in range(1, len(x_h)))
        #     # hist = hist * f / len(nodes)
        #     # max_h = np.max(hist)
        #     # hist = hist / max_h
        #     # plt.plot(x_h, hist,
        #     #          linewidth=1, label='density (max {})'.format(max_h))
        #
        #     v = self.recursive_traversal(func=(lambda node: node.get_value()),
        #                                  collect_cond_func=lambda node: node.is_expandable())
        #     x = sorted(self.recursive_traversal(func=(lambda node: node.get_location()),
        #                                         collect_cond_func=lambda node: node.is_expandable()))
        #     ev = self._expand_threshold_value(1) - .5
        #     max_v = np.max(v)
        #     if max_v != 0:
        #         plt.plot(x, v /
        #                  np.max(v) - 1, label='values (max {})'.format(max_v))
        #         plt.plot([x[0], x[len(x) - 1]], [ev / np.max(v) - 1] * 2,
        #                  label='expansion threshold = {}(f={})'.format(ev + 0.5, self._get_mean_distance() / self._get_max_mean_distance()), linewidth=0.8)

        # plt.legend()
        plt.grid(True)
        plt.xlim(-.05, 1.05)
        if save:
            plt.savefig("{}/a{}.png".format(path, self.SAVE_ID))
            self.SAVE_ID += 1
        else:
            plt.show()
        # plt.gcf().clear()

    @staticmethod
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


if __name__ == '__main__':
    # from bin_exp_test import test, test2
    # test()
    dims = 1
    tree_size = 127
    iterations = 100
    max_size = 20
    tree = Exploration_tree(dims, tree_size)
    tree.plot()
    exit()
    # samples_size_buffer = np.random.random(iterations) * max_size + 10
    # samples_size_buffer = samples_size_buffer.astype(int)
    #
    # samples = None
    # count = 0
    # for i in samples_size_buffer:
    #     print(count, '----new iteration, searches', i)
    #     count += 1
    #
    #     center = 0.1  # if count < 40 else .7
    #     samples = center + np.abs(np.random.standard_normal((i, dims))) * 0.05
    #     starting_size = tree.get_current_size()
    #     for p in samples:
    #         p = list(p)
    #         tree.search_nearest_node(p)
    #
    #     # ending_size = tree.get_size()
    #     # # print('added', i, 'points(', samples, '): size before-after', starting_size,
    #     # #       '-', ending_size, '({})'.format(ending_size - starting_size))
    #     # if starting_size + i != ending_size:
    #     #     print('ERROR')
    #     #     tree.plot()
    #     #     exit()
    #     tree.plot()
    #
    #     tree.update()
    #     # tree.plot()
    #
    #     # exit()
    # # tree.plot()
