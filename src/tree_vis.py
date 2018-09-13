#!/usr/bin/python3
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D


import numpy as np

from ntree import *

from node import *


def average_timeline(x):
    res = []
    count = 0
    total = 0
    for i in x:
        total += i
        count += 1
        res.append(total / count)
    return res


def apply_func_to_window(data, window_size, func):
    data_lenght = len(data)
    window_size = min(window_size, data_lenght)
    if window_size == 0:
        window_size = (data_lenght * .1)
    res = []
    for i in range(data_lenght):
        start = int(max(i - window_size / 2, 0))
        end = int(min(i + window_size / 2, data_lenght - 1))
        if start == end:
            continue
        res.append(func(data[start:end]))

    return res


def break_into_batches(array, number_of_batches=-1, size_of_batches=1):
    if number_of_batches < 0:
        number_of_batches = len(array)
    number_of_batches = max(1, min(number_of_batches, len(array)))
    size_of_batches = max(1, min(size_of_batches, len(array)))

    array = np.array(array)

    ranges = np.linspace(0, len(array), number_of_batches + 1)
    indexes = list((ranges[i] + ranges[i - 1]) / 2 for i in range(1, len(ranges)))

    res = []

    for index in indexes:
        start = index - size_of_batches / 2
        end = start + size_of_batches
        # int
        start = max(0, int(round(start)))
        end = min(len(array), int(round(end)))

        res.append(array[range(start, end)])
    return res


def plot(tree, save=False, path='/home/jim/Desktop/temp_result_pics/a.png', red_levels=False):

    dims = tree._dimensions

    if dims == 1:
        plot_1d_tree(tree, red_levels=red_levels)
    elif dims == 2:
        plot_2d_tree(tree, red_levels=red_levels)
    elif dims == 3:
        plot_3d_tree(tree, red_levels=red_levels)
    else:
        print("plot works for 3 or less dimensional trees")

    if save:
        plt.savefig(path, dpi=300)
        np.savetxt(path[:len(path) - 4] + '.txt', tree.get_points())
    else:
        plt.show()


def plot_point_density(tree, save=False, path='/home/jim/Desktop/'):

    dims = tree._dimensions

    if dims == 1:
        # plot_1dpoint_hist(tree)
        plot_1d_point_density(tree)
    elif dims == 2:
        plot_2dpoint_hist(tree)
        # plot_2d_tree(tree)
    else:
        print("plot_point_density works for 2 or less dimensional trees")

    if save:
        plt.savefig("{}/plot_point_density{}.png".format(path, SAVE_ID))
        SAVE_ID += 1
    else:
        plt.show()


def plot_values(tree, save=False, path='/home/jim/Desktop/'):
    dims = tree._dimensions

    if dims == 1:
        plot_values_1d(tree)
    elif dims == 2:
        plot_values_2d(tree)
        pass
    else:
        print("plot_values works for 2 or less dimensional trees")

    if save:
        plt.savefig("{}/plot_point_density{}.png".format(path, SAVE_ID))
        SAVE_ID += 1
    else:
        plt.show()


def plot_1d_tree(tree, red_levels=False):
    plt.figure()
    plt.title('tree size = {}'.format(tree.get_current_size()))

    nodes = tree.get_nodes(recalculate=True)

    max_level = np.max(list(node.get_level() for node in nodes))

    node_to_point = (lambda node: [node.get_location()[0], node.get_level()])

    for node in nodes:
        child = node_to_point(node)
        parent = node_to_point(node._parent) if node._parent is not None else child

        x = [child[0], parent[0]]
        y = [child[1], parent[1]]

        plt.plot(x, y, 'm-', linewidth=0.2)
        size = np.interp(node.get_level(), [1, max_level], [15, 2])
        if red_levels:
            color = '#{:02x}0000'.format(
                round(np.interp(node.get_level(), [0, max_level], [0, 255])))
        else:
            color = 'C{}'.format(
                round(np.interp(node.get_level(), [0, max_level], [0, min(max_level, 9)])))

        plt.plot(x[0], y[0],
                 color, marker='.', markersize=size)

        plt.plot([x[0]] * 2, [-0.1, -0.2],
                 color, linewidth=0.5)
    plt.grid(True)
    plt.xlim(-.05, 1.05)
    plt.yticks(range(max_level + 1))


def plot_2d_tree(tree, add_3d=True, red_levels=False):

    nodes = tree.get_nodes()

    max_level = np.max(list(node.get_level() for node in nodes))

    fig = plt.figure()

    if add_3d:
        ax1 = fig.add_subplot(121)

    node_to_point = (lambda node: node.get_location())

    for node in nodes:
        child = node_to_point(node)
        parent = node_to_point(node._parent) if node._parent is not None else child

        x = [child[0], parent[0]]
        y = [child[1], parent[1]]

        plt.plot(x, y, 'm-', linewidth=0.2)

        size = np.interp(node.get_level(), [1, max_level], [15, 2])
        if red_levels:
            color = '#{:02x}0000'.format(
                round(np.interp(node.get_level(), [0, max_level], [0, 255])))
        else:
            color = 'C{}'.format(
                round(np.interp(node.get_level(), [0, max_level], [0, min(max_level, 9)])))
        plt.plot(x[0], y[0],
                 color, marker='.', markersize=size)

        plt.plot([x[0]] * 2, [-0.01, -0.05],
                 '#000000', linewidth=0.5)
        plt.plot([-0.01, -0.05], [y[0]] * 2,
                 '#000000', linewidth=0.5)

    plt.grid(True)
    plt.xlim(-.05, 1.05)
    plt.ylim(-.06, 1.05)

    if add_3d:
        ax2 = fig.add_subplot(122, projection='3d')
        for node in nodes:
            # add the depth of the nodes as a 3d axis
            point = np.append(node.get_location(), node.get_level())
            if node.is_root():
                parent_point = point
            else:
                parent = node._parent
                parent_point = np.append(parent.get_location(), parent.get_level())

            ax2.plot([point[0], parent_point[0]],
                     [point[1], parent_point[1]],
                     [point[2], parent_point[2]], c='m', linewidth=.4)

            size = np.interp(node.get_level(), [1, max_level], [15, 2])

            if red_levels:
                color = '#{:02x}0000'.format(
                    round(np.interp(node.get_level(), [0, max_level], [0, 255])))
            else:
                color = 'C{}'.format(
                    round(np.interp(node.get_level(), [0, max_level], [0, min(max_level, 9)])))
            ax2.scatter(point[0], point[1], point[2], color=color, s=size)

        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('level')
        ax2.zaxis.set_major_locator(MaxNLocator(integer=True))

        ax2.view_init(30, 5)


def plot_3d_tree(tree, red_levels=False):
    nodes = tree.get_nodes()
    max_level = np.max(list(node.get_level() for node in nodes))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'b']

    for node in nodes:
        # add the depth of the nodes as a 3d axis
        point = node.get_location()
        level = int(node.get_level())
        if node.is_root():
            parent_point = point
        else:
            parent = node._parent
            parent_point = parent.get_location()

        ax.plot([point[0], parent_point[0]],
                [point[1], parent_point[1]],
                [point[2], parent_point[2]], c='m', linewidth=.4)

        size = np.interp(node.get_level(), [1, max_level], [15, 2])
        if red_levels:
            color = '#{:02x}0000'.format(
                round(np.interp(node.get_level(), [0, max_level], [0, 255])))
        else:
            color = 'C{}'.format(
                round(np.interp(node.get_level(), [0, max_level], [0, min(max_level, 9)])))
        ax.scatter(point[0], point[1], point[2], color=color, s=size)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(40, 10)


def plot_1d_point_density(tree, resolution=.05):
    nodes = tree.get_nodes()
    points = np.sort(list(node.get_location()[0] for node in nodes))
    x = np.linspace(0, 1, 1001)
    # x = np.copy(points)
    # x = np.insert(x, 0, 0)
    # x = np.insert(x, len(x), 1)
    num_of_neighbors = []
    ranges = []
    for point in x:
        range = (resolution / 2 if point - resolution / 2 > 0 else point) \
            + (resolution / 2 if point + resolution / 2 <= 1 else 1 - point)
        ranges.append(range)
        neighbors = np.where(np.abs(points - point) < resolution / 2)[0]

        num_of_neighbors.append(len(neighbors))
    ranges = np.array(ranges)
    num_of_neighbors = np.array(num_of_neighbors)
    density = np.divide(num_of_neighbors, ranges)
    plt.plot(x,  apply_func_to_window(density, 10, np.average))
    # # dists = list(np.average(list(batch[i + 1] - batch[i] for i in range(len(batch) - 1)))
    # #               for batch in break_into_batches(points, number_of_batches=len(points), size_of_batches=int(len(points) * 0.1)))
    # # density = list(1 / dist for dist in dists)
    # # plt.plot(points, density, label='density (nodes/unit)')
    # value_lambdas = [lambda node: node.get_location()[0],
    #                  lambda node: node.get_level()]
    # #, lambda node: node.get_value_increase_if_cut()]
    #
    # infos = list(list(l(node) for l in value_lambdas) for node in nodes)
    #
    # infos = sorted(infos, key=lambda _: _[0])
    # levels = list(item[1] for item in infos)
    # levels = list(np.max(batch) for batch in break_into_batches(levels, -1, int(len(levels) / 10)))
    # plt.plot(points, levels, label='levels')
    #
    # plt.hist(points, bins='auto', histtype='step', label='hist')
    # for point in points:
    #     plt.plot([point, point], [0, 0.1 * np.max(density)], 'm', linewidth=.5, label='points')
    plt.plot(points, np.zeros(len(points)), 'm^')

    plt.xlabel('Space')
    plt.ylabel('Nodes/{}'.format(resolution))
    plt.title('Density')
    plt.grid(True)
    # plt.legend()


def plot_1dpoint_hist(tree):
    nodes = tree.get_nodes()
    points = list(node.get_location()[0] for node in nodes)

    plt.hist(points, bins=int(.1 * len(points)))
    plt.grid(True)
    plt.legend()


def plot_2dpoint_hist(tree):
    nodes = tree.get_nodes()
    points = list(node.get_location() for node in nodes)
    xs = list(point[0] for point in points)
    ys = list(point[1] for point in points)

    plt.hist2d(xs, ys)
    plt.colorbar()

    plt.grid(True)
    plt.legend()


def plot_values_1d(tree):
    nodes = tree.get_nodes()

    value_lambdas = [lambda node: node.get_location()[0],
                     lambda node: node.get_value()]
    #, lambda node: node.get_value_increase_if_cut()]

    infos = list(list(l(node) for l in value_lambdas) for node in nodes)

    infos = sorted(infos, key=lambda _: _[0])

    x = list(item[0] for item in infos)
    ys = []
    for i in range(1, len(value_lambdas)):
        values = list(item[i] for item in infos)
        plt.plot(x, values, '1-')

    plt.plot([0, 1], [tree.get_mean_value()] * 2)

    plt.grid(True)
    plt.legend()


def plot_values_2d(tree):
    from operator import itemgetter

    nodes = tree.get_nodes()
    points = list((node.get_location()[0],
                   node.get_location()[1],
                   node.get_value()) for node in nodes)
    # points = sorted(points, key=itemgetter(1, 2))
    #
    X = list(item[0] for item in points)
    Y = list(item[1] for item in points)
    Z = list(item[2] for item in points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Grab some test data.

    # Plot a basic wireframe.
    ax.scatter(X, Y, Z)
    plt.show()


def plot_nodes_1d(nodes,
                  node_x=(lambda node: node.get_location()[0]),
                  node_y=(lambda node: node.get_value()),
                  marker='.', label=None):

    x = []
    y = []
    for node in nodes:
        x.append(node_x(node))
        y.append(node_y(node))

    for i in range(len(x)):
        plt.plot(x[i], y[i], marker, label=label)


def pre_update_plot(tree):

    max_level = np.max(list(node.get_level() for node in tree.get_nodes()))

    plot_nodes_1d(tree.get_nodes(),
                  marker='b.', label='values')
    plot_nodes_1d(tree.get_expendable_nodes(),
                  marker='c1', label='expanable')

    to_prune = tree.prune_prospectives()
    plot_nodes_1d(to_prune,
                  node_x=(lambda node: [node.get_location()[0]] * 2),
                  node_y=(lambda node: [node.get_value(), node.get_value() +
                                        node.get_value_increase_if_cut()]),
                  marker='r2--', label='to prune ({})'.format(len(to_prune)))
    plot_nodes_1d(tree.get_nodes(),
                  node_y=(lambda node: node.get_level() / max_level - 1.2),
                  marker='g.', label='tree (size {})'.format(tree.get_current_size()))
    # plot_nodes_1d(tree.get_expendable_nodes(),
    #               node_y=lambda node: node.get_value_increase_if_cut(),
    #               marker='r3', label='value incr if cut')

    # plot_nodes_1d(tree.get_nodes(),
    #               node_x=lambda node: [
    #                   node.get_location()[0], node.suggest_for_expand()[0].get_location()[0]],
    #               node_y=lambda node: [
    #                   node.get_level() / max_level - 1.2, node.suggest_for_expand()[0].get_level() / max_level - 1.2],
    #               marker='m--')

    plt.plot([0, 1], [tree.get_mean_value()] * 2)

    plt.grid(True)
    plt.legend()

    plt.show()
