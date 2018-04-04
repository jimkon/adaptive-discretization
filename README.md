# Problem description
Discretizating a space with a fixed number of points is usually trivial. Most of the times a uniform distribution fits our needs because there is no specific interest on any range inside the given space. That actually means that the uniformly distributed points we assigned will match the uniform distribution of the **continuous** points that we need. So when we search the nearest neighbor of a given continuous point, the **mean distance** of the corresponding discrete point will be the minimum.

Nevertheless, the are cases that inside the given range, there are some places with higher interest. These places can be specific ranges that we search inside more often which make the uniform discretization not optimal. An increased resolution in this range would give us a lower average distance on the searches. Another case is when we actually need specific points in the space. In this case, if these points are less than out discrete ones, we could even bring our average distance to zero if we could adapt our discretization. 

# My solution
In this library, i propose a solution that makes possible the adaption of the discretized points with an automated way. The goal is to minimize the mean distance for every kind of distribution while remaining a steady number of discrete points.

Trying to achieve:
1.  Minimum average distance for any distribution   
    * Provide the best posible precision
    * Adapting to uknownn to the user distributions 
2.  Stable number of discrete points
3.  Work for any dimensional space
4.  Time complexity between O(logN) and O(N) for search, insert and delete


# Architecture
Like the most related approaches, i used trees similar to kd-trees to solve this problem. Unlike kd-trees that are binary, i break each axis into 2 subranges, which bring a branch factor equal to 2

