# Problem description
Discretizating a space with a fixed number of points is usually trivial. Most of the times a uniform distribution fits our needs because there is no specific interest on any range inside the given space. That actually means that the uniformly distributed points we assigned will match the uniform distribution of the **continuous** points that we need. So when we search the nearest neighbor of a given continuous point, the **mean distance** of the corresponding discrete point will be the minimum.

Nevertheless, the are cases that inside the given range, there are some places with higher interest. These places can be specific ranges that we search inside more often which make the uniform discretization not optimal. An increased resolution in this range would give us a lower average distance on the searches. Another case is when we actually need specific points in the space. In this case, if these points are less than out discrete ones, we could even bring our average distance to zero if we could adapt our discretization.

# My solution
In this library, i propose a solution that makes possible the adaption of the discretized points with an automated way. The goal is to minimize the mean distance for every kind of distribution while remaining a steady number of discrete points.

Trying to achieve:
1.  Minimum Mean Error ($ME$) for any distribution.
    * Provide the best possible precision.
    * Adapting to unknown to the user distributions.
2.  Stable number of discrete points(size $K$).
3.  Work for any dimensional space(number of dimensions $n$).
4.  Sub-linear complexity for search, insert and delete.

**Note:** The nearest neighbor search is not a feature of this solution, but the algorithm can easily be combined with another package for this, like [FLANN](https://github.com/mariusmuja/flann).


# Architecture
Like the most related approaches, i used trees similar to kd-trees to solve this problem. Unlike kd-trees that have branching factor = $2$, this tree splits each axis into $2$ which gives a branching factor equal to branching factor = $2^n$. Each node represents a discrete point in the middle of an area that is assigned to it. By extending nodes, the precision is increasing.  Also nodes with located in areas out of interest get cut in order to maintain the number of discrete points stable. With this procedure the tree tries to adapt to the distribution of the points that user is searching.

## Initialization
The starting tree is uniform and full grown until the level that brings us closest and below to the requested size. The size of the the full grown tree until a given level is computed by the following formula:

$size = \sum_{i=0}^{level}{2^{i*n}}$

Some examples:    

|dims\level| 1 | 2 | 3 | 4 | 5 |   
| --- | :---: | :---: | :---: | :---: | :---: |
|1|1| 3| 7| 15| 31|
|2|1| 5| 21| 85| 341|
|3|1| 9| 73| 585| 4681|
|4|1| 17| 273| 4369| 69905|
|5|1| 33| 1057| 33825| 1082401|

## Adaption
Each node keeps a record of the total error it collected from the searches in its assigned area. The average of the total error of each node is the $ME$. This algorithm tries to find a delicate balance between precision and size in order to give the minimum $ME$. The following two competitive mechanisms are applied repeatedly for this purpose until the balance is achieved.

*   #### Expanding nodes
    Nodes with high total error are the first candidates of expanding. Expanding those nodes will add $2^n$ new different nodes on this area. After that the total error on this area drops a reasonable [amount](link to expansion limits). The rate of the adaption is same as a binary search, something that minimize the loss of the adapting procedure.  

*   #### Cutting branches
    Because it is very important to keep a pretty stable size, nodes that contribute the least, has to be cut to maintain the right size and to make space for more expansions to be done. Usually, nodes with low total error values are the ones that need cut. But selecting a node to be cut is not as straightforward as it seems to be. Simply cutting those with the lowest values is not optimal because usually the reason of this low value is an expansion made earlier to reduce a higher one. Cutting this node will free will lead to a very high value that will be expanded right after and this will happen repeatedly until the end. So we have to be careful about what results this cut is going to bring.
