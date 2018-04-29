#!/usr/bin/python3

import numpy as np


class Node:

    BRANCH_MATRIX = None

    def __init__(self, parent, direction, dims=-1):
        self._parent = parent

        if parent is None:
            self._radius = .5
            self._location = np.ones(dims) * 0.5
            self._level = 0
            self._value = 0
        else:
            self._radius = parent._radius / 2
            self._location = parent.get_location() + direction * self._radius
            self._level = parent._level + 1
            self._value = parent._value

        self._low_limit = self._location - self._radius
        self._high_limit = self._location + self._radius

        if self.BRANCH_MATRIX is None:
            self._init_branch_matrix(len(self._location))

        self._branches = [None] * len(self.BRANCH_MATRIX)
        self._value_without_branch = np.zeros(len(self.BRANCH_MATRIX))

        self.__achieved_precision_limit = False

        if np.array_equal(self._low_limit, self._high_limit):
            self.__achieved_precision_limit = True
            # raise ArithmeticError('Node: Low == High :{}=={}'.format(
            #     self._low_limit, self._high_limit))

    def expand(self, towards_point=None):
        if towards_point is None:
            towards_point = self.get_location()

        new_nodes = []
        for i in self._indexes_of_relevant_branches(towards_point):
            if self._branches[i] is not None:
                continue
            new_node = Node(self, self.BRANCH_MATRIX[i])
            self._branches[i] = new_node
            new_nodes.append(new_node)

        return new_nodes

    def expand_suggestions(self):
        new_nodes = []
        for to_expand in self.suggest_for_expand():
            new_nodes.extend(to_expand.expand(self.get_location()))
        return new_nodes

    def search(self, point, min_dist_till_now=1, increase_value=True):
        if not self._covers_point(point):
            return None, 0

        dist_to_self = np.linalg.norm(self._location - point)

        if min_dist_till_now > dist_to_self:
            min_dist_till_now = dist_to_self

        branches = self._branches
        for branch_i in self._indexes_of_relevant_branches(point):
            branch = branches[branch_i]
            if branch is None:
                continue
            res, branch_dist = branch.search(point, min_dist_till_now, increase_value)
            if res is not None:
                if branch_dist > dist_to_self:
                    if increase_value:
                        self._value += dist_to_self
                    return res, dist_to_self
                else:
                    if increase_value:
                        self._value_without_branch[branch_i] += dist_to_self
                    return res, branch_dist

        if increase_value:
            self._value += dist_to_self if min_dist_till_now == dist_to_self else 0
        return self, dist_to_self

    def delete(self):
        if self.is_root():
            return
        index = self._parent._indexes_of_relevant_branches(self.get_location())[0]
        self._parent._branches[index] = None
        # self._parent._value += self._parent._value_without_branch[index]

    def recursive_collection(self, result_array, func, traverse_cond_func, collect_cond_func):
        if not traverse_cond_func(self):
            return
        if collect_cond_func(self):
            result_array.append(func(self))
        for branch in self.get_branches():
            branch.recursive_collection(result_array, func, traverse_cond_func, collect_cond_func)

    def get_value(self):
        return self._value

    def get_value_increase_if_cut(self):
        if self.is_root() or not self.is_expandable():
            return 0

        temp = self._parent._value_without_branch[self._parent._branches.index(self)]

        return temp - self.get_value()

    def suggest_for_expand(self):
        res = []
        if self.is_expandable():
            res.append(self)
        res.extend((branch.search(self.get_location(), increase_value=False)
                    [0] for branch in self.get_branches()))
        # print('res', res)
        return res

    def reset_value(self):
        self._value = 0
        self._value_without_branch = np.zeros(len(self.BRANCH_MATRIX))

    def get_level(self):
        return self._level

    def get_location(self):
        return self._location

    def get_branches(self):
        res = []
        for branch in self._branches:
            if branch is not None:
                res.append(branch)
        return res

    def number_of_childs(self):
        return len(self.get_branches())

    def is_root(self):
        return self._parent is None

    def is_leaf(self):
        return len(self.get_branches()) == 0

    def is_expandable(self):
        return self.number_of_childs() < len(self.BRANCH_MATRIX) or self.__achieved_precision_limit

    def _direction_matrix_for_(self, point):
        assert len(point) == len(self.get_location()), 'points must have same lenght'
        p0 = self.get_location()
        sub = point - p0
        # if any sub[i] == 0
        for i in range(len(sub)):
            if sub[i] == 0:
                p_mask = np.zeros(len(sub))
                p_mask[i] = 1
                res = self._direction_matrix_for_(point + p_mask)
                res.extend(self._direction_matrix_for_(point - p_mask))
                return res

        norm_sub = [sub / np.abs(sub)]
        return norm_sub

    def _indexes_of_relevant_branches(self, p):
        dirs = self._direction_matrix_for_(p)

        indexes = []

        for dir in dirs:
            to_bin = (dir + 1) / 2
            to_index = list(to_bin[i] * (2**(len(dir) - i - 1))
                            for i in range(len(dir)))
            index = int(np.sum(to_index))
            indexes.append(index)

        return indexes

    def _covers_point(self, point):
        check1 = self.point_less_or_equal_than_point(self._low_limit, point)
        check2 = self.point_less_or_equal_than_point(point, self._high_limit)
        return check1 and check2

    def _equals(self, node):
        return np.array_equal(self.get_location(), node.get_location())

    def __str__(self):
        return 'loc={} level={} r={} br={} v={} parent_loc={}'.format(self._location,
                                                                      self._level,
                                                                      self._radius,
                                                                      len(self.get_branches()),
                                                                      self._value,
                                                                      self._parent.get_location() if self._parent is not None else None)

    __repr__ = __str__

    @staticmethod
    def point_less_or_equal_than_point(point1, point2):
        assert len(point1) == len(point2), 'points must have same lenght'
        for i in range(len(point1)):
            if point1[i] > point2[i]:
                return False
        return True

    @staticmethod
    def _init_branch_matrix(dims):
        import itertools
        Node.BRANCH_MATRIX = np.array(list(itertools.product([0, 1], repeat=dims))) * 2 - 1
