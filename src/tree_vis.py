#!/usr/bin/python3
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D


import numpy as np

from ntree import *

from node import *


SAVE_ID = 0


def plot(tree, save=False, path='/home/jim/Desktop/dip/Adaptation-of-Action-Space-for-Reinforcement-Learning/results/pics'):

    dims = tree._dimensions

    if dims == 1:
        plot_1d_tree(tree)
    if dims == 2:
        plot_2d_tree(tree)
    if dims == 3:
        plot_3d_tree(tree)

    if save:
        plt.savefig("{}/plot{}.png".format(path, SAVE_ID))
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
