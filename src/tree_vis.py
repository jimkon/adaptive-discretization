#!/usr/bin/python3
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D


import numpy as np

from ntree import *

from node import *


SAVE_ID = 0


def plot(tree, save=False, path='/home/jim/Desktop/'):

    dims = tree._dimensions

    if dims == 1:
        plot_1d_tree(tree)
    elif dims == 2:
        plot_2d_tree(tree)
    elif dims == 3:
        plot_3d_tree(tree)
    else:
        print("plot works for 3 or less dimensional trees")

    if save:
        plt.savefig("{}/plot{}.png".format(path, SAVE_ID))
        SAVE_ID += 1
    else:
        plt.show()


def plot_point_density(tree, save=False, path='/home/jim/Desktop/'):

    dims = tree._dimensions

    if dims == 1:
        plot_1dpoint_hist(tree)
        # plot_1d_point_density(tree)
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
        pass
    else:
        print("plot_values works for 2 or less dimensional trees")

    if save:
        plt.savefig("{}/plot_point_density{}.png".format(path, SAVE_ID))
        SAVE_ID += 1
    else:
        plt.show()


def plot_1d_tree(tree, node_to_point=(lambda node: node.get_location())):
    plt.figure()
    plt.title('tree size = {}'.format(tree.get_current_size()))

    nodes = tree.get_nodes()

    max_level = np.max(list(node.get_level() for node in nodes))

    node_to_point = (lambda node: [node.get_location()[0], node.get_level()])

    for node in nodes:
        child = node_to_point(node)
        parent = node_to_point(node._parent) if node._parent is not None else child

        x = [child[0], parent[0]]
        y = [child[1], parent[1]]

        plt.plot(x, y, 'm-', linewidth=0.2)
        size = np.interp(node.get_level(), [1, max_level], [15, 2])
        color = 'C{}'.format(
            round(np.interp(node.get_level(), [0, max_level], [0, min(max_level, 9)])))
        plt.plot(x[0], y[0],
                 color, marker='.', markersize=size)

        plt.plot([x[0]] * 2, [-0.1, -0.2],
                 color, linewidth=0.5)
    plt.grid(True)
    plt.xlim(-.05, 1.05)
    plt.yticks(range(max_level + 1))


def plot_2d_tree(tree, add_3d=True):

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

            color = 'C{}'.format(
                round(np.interp(node.get_level(), [0, max_level], [0, min(max_level, 9)])))
            ax2.scatter(point[0], point[1], point[2], color=color, s=size)

        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('level')
        ax2.zaxis.set_major_locator(MaxNLocator(integer=True))

        ax2.view_init(30, 5)


def plot_3d_tree(tree, node_to_point=(lambda node: node.get_location())):
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


def plot_1d_point_density(tree):
    nodes = tree.get_nodes()
    points = np.sort(list(node.get_location()[0] for node in nodes))
    density = []

    for i in range(len(points)):
        prev = points[i - 1] if i > 0 else points[i + 1]
        curr = points[i]
        next = points[i + 1] if i < len(points) - 1 else points[i - 1]
        density.append((abs(curr - prev) + abs(next - curr)) / 2)

    density = apply_func_to_window(1 / np.array(density), int(.1 * len(density)), np.average)

    plt.plot(points, density, label='density (nodes/unit)')
    plt.grid(True)
    plt.legend()


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

    plt.hist2d(xs, ys, bins=int(.1 * len(points)))
    plt.colorbar()

    plt.grid(True)
    plt.legend()


def plot_values_1d(tree):
    nodes = tree.get_nodes()

    value_lambdas = [lambda node: node.get_location()[0],
                     lambda node: node.get_value(),
                     lambda node: node.get_value_increase_if_cut()]

    infos = list(list(l(node) for l in value_lambdas) for node in nodes)

    infos = sorted(infos, key=lambda _: _[0])

    x = list(item[0] for item in infos)
    ys = []
    for i in range(1, len(value_lambdas)):
        values = list(item[i] for item in infos)
        plt.plot(x, values, '1-')

    plt.grid(True)
    plt.legend()


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


tree = Tree(1, 31)
tree._nodes.extend(tree._root.expand_rec([0]))
tree._nodes.extend(tree._root.expand_rec([1]))
# tree._nodes.extend(tree._root._branches[0].expand_rec([0.1]))
# tree.plot()


batches = 10
batch_size = 10000
x = np.linspace(0, 1, 101)

for i in range(batches):
    # samples = np.random.choice(x, size=batch_size)
    samples = np.linspace(0.2, 1, 10000)
    for sample in samples:
        tree.search_nearest_node([sample])

    print(tree._get_mean_distance())
    plot_values(tree)
    plot(tree)
    exit()

    if i != batches - 1:
        tree.update()

plot(tree)
