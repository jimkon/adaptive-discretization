# Install
pip install git+https://github.com/jimkon/adaptive-discretization

# Examples
[Notebooks](https://jimkon.github.io/adaptive-discretization/)

# Problem description
Discretizing a space with a fixed number of points is usually trivial. Most of the time a uniform distribution fits our needs because there is no specific interest in any range inside the given space. That actually means that the uniformly distributed points we assign will match the uniform distribution of the **continuous** points that we need. So when we search the nearest neighbour of a given continuous point, the **mean distance** of the corresponding discrete point will be the minimum.

Nevertheless, sometimes there are some places inside the given range with higher interest. These places can be specific ranges or points that we need more often which makes the uniform discretization not optimal. An increased resolution in this range would give us a lower average distance on the searches.

This raises some questions though. Where to increase the resolution and how much? And if we increase the resolution in a region, it's necessary to decrease the resolution in another region in order to keep the number of points stable.


# My solution (a brief explanation)


In this library, I propose a solution that makes possible the adaption of the discretized points in an automated way. The goal is to minimize the mean distance for every kind of distribution while maintaining a steady number of discrete points.

Trying to achieve:
1.  Minimum Mean Error (_ME_) for any distribution.
    * Provide the best possible precision.
    * Adapting to unknown to the user distributions.
2.  Stable number of discrete points(size _K_).
3.  Work for any dimensional space(number of dimensions __n__).
4.  Sub-linear complexity for search, insert and delete.

**Note:** The nearest neighbour search is not a feature of this solution, but the algorithm can easily be combined with another package for this, like [FLANN](https://github.com/mariusmuja/flann).


## Architecture
Like the other related approaches, I used trees to solve this problem. Each node represents a discrete point in the middle of an area that is assigned to it. By extending nodes, the precision is increasing.  Also, nodes located in areas out of interest get cut in order to maintain the number of discrete points. User's "searches" are taken as feedback in order to evaluate the current discrete points. With this procedure, the tree tries to adapt to the distribution of the points that the user is searching.

Another thing that is worth mentioning is that this architecture is set to work with n-dimensional __unit__ cubes. In other words, the ranges of the points in each direction are fixed to (0, 1). The components of each point searched are forced in this range. The reason for this is that it was way easier to implement with a fixed range rather than caring about all the different ranges for each axis. But it is very easy to convert this range to any other with some linear transformations.

#### Initialization
The starting tree is uniform and fully grown until the level that brings us closest and below the requested size. In the next few steps, the tree will expand to the requested size. The loss that comes from this procedure is very little because the expansion is very quick.  However, it is a good idea to choose a size close to one of the initial sizes to minimize this.




#### Adaption
Each node keeps a point and a record of the total distance from this point and the searches in its assigned area. The average of the total error of each node is the _ME_. This algorithm tries to find a delicate balance between precision and size in order to give the minimum _ME_. The following two competitive mechanisms are applied repeatedly for this purpose until the balance is achieved.

*   ##### Cutting branches
    Because it is very important to keep a pretty stable size, nodes that contribute the least have to be cut to maintain the right size and to make space for more expansions to be done. Usually, nodes with low total error values are the ones that need cut. But simply cutting those with the lowest values is not optimal because usually, the reason for this low value is an expansion made earlier to reduce a higher one. Before cutting these nodes we first have to check how much they will increase the _ME_.

*   ##### Expanding nodes
    Nodes with high total error are the first candidates for expansion. Expanding those nodes will add 2^n new different nodes in this area. After that, the total error in this area drops a reasonable [amount](link to expansion limits). The rate of the adaption is the same as a binary search, something that minimizes the loss of the adapting procedure.  






