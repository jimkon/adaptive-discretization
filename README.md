# Problem description
Discretizating a space with a fixed number of points is usually trivial. Most of the times a uniform distribution fits our needs because there is no specific interest on any range inside the given space. That actually means that the uniformly distributed points we assigned will match the uniform distribution of the **continuous** points that we need. So when we search the nearest neighbor of a given continuous point, the **mean distance** of the corresponding discrete point will be the minimum.

Nevertheless, the are cases that inside the given range, there are some places with higher interest. These places can be specific ranges that we search inside more often which make the uniform discretization not optimal. An increased resolution in this range would give us a lower average distance on the searches. Another case is when we actually need specific points in the space. In this case, if these points are less than out discrete ones, we could even bring our average distance to zero if we could adapt our discretization.

# My solution
In this library, i propose a solution that makes possible the adaption of the discretized points with an automated way. The goal is to minimize the mean distance for every kind of distribution while remaining a steady number of discrete points.

Trying to achieve:
1.  Minimum Mean Error(__ME__) for any distribution   
    * Provide the best possible precision
    * Adapting to unknown to the user distributions
2.  Stable number of discrete points(size __K__)
3.  Work for any dimensional space(number of dimensions __N__)
4.  Sub-linear complexity for search, insert and delete

**Note:** The nearest neighbor search is not a feature of this solution, but the algorithm can easily be combined with another package for this, like [FLANN](https://github.com/mariusmuja/flann).


# Architecture
Like the most related approaches, i used trees similar to kd-trees to solve this problem. Each node represents a discrete point in the middle of a fixed range. By extending nodes, the depth is increasing, and the precision is increasing too.  Also nodes with points out of the range of interest get cut in order to maintain the number of discrete points stable. With this procedure the tree tries to adapt to the distribution of the points that user is searching.

## Initialization
Initially, the tree reaches the closest

## Adaption

## Expansion
Expanding a nodes gives a greater level of precision. Unlike kd-trees that are binary, i break each axis into 2 subranges, which gives a branching factor equal to 2^_n_, where _n_ is the number of dimensions.
