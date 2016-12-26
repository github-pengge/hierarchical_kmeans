An package for creating hierarchical k-means/k-means tree/vocabulary tree.

## Hierarchical k-means
An implementation of k-means tree, it's a kind of hierarchical k-means algorithm. 

As we know, doing k-means on large dataset is really time consuming. There are many approximation algorithm. Hierarchical k-means is one of them.

This code implements a tree-like k-means. Here is an example:

Assume we are doing a two-level tree k-means, the first level is a 20-class k-means, and for each second level branch, a 10-class k-means is done. Then, we have done a k-means with totally `20*10=200` classes. When quantizing a data point, we first quantizing at first level, that is, projecting the data point to one of the 20 classes. After that, quantizing at the second level at the nearest branch.

## Requirements
This implementation is based on the implementation of `k-means` from [`scipy`](https://www.scipy.org/) package. Make sure you had installed [`scipy`](https://www.scipy.org/) successfully.

## Usage
To do hierarchical k-means, we should first determine the structure of hierarchical k-means. This is an example:
``` python
from hierarchical_kmeans import tree
cluster_structure = tree('my_cluster_structure')
for i in range(100):
    x = tree('l1-c%d' % (i+1))
    cluster_structure.add_child(x)
    for j in range(50):
        y = tree('l2-c%d-c%d' % (i+1, j+1))
        x.add_child(y)
        for k in range(20):
            z = tree('l3-c%d-c%d-c%d' % (i+1, j+1, k+1))
            y.add_child(z)
```
This code create a tree with 3 levels, it tells the constructor(when passing it to the hierarchical k-means constructor) to create a
k-means tree with depth-3: for first level, doing a 100-class k-means; for each
branch(100 branches) in level 2, doing a 50-class k-means; and for each branch in level 3, doing a 20-class k-means. Note that if you need a 100-class-clustering, k-means will not always get the exactly 100 classes, some class may be empty, so commonly speaking, the example given above will get a total clustering of less than `100*50*20` classes, but I promise this is not a problem.

After creating cluster structure, doing hierarchical k-means is simple:
``` python
from hierarchical_kmeans import hierarchical_kmeans as hkmeans
hk = hkmeans(cluster_structure)
hk.cluster(your_row_base_data, only_store_id=True, iteration=1)
```
Note that `hk.cluster` function accepts three arguments, the first argument is your data(row-based), the second argument `only_store_id` tells the algorithm to store the data id(`only_store_id=True`, default for saving memory) or the data(`only_store_id=False`) on the leaf nodes. Note that the data id(or data) is store in `additional_data` attribute of leaf nodes. The third argument `iteration` inherits from `kmeans` function from `scipy.cluster.vq`, when `iteration > 1`, for each branch, it chooses the best kmeans result(with lowest distortion) in `iteration` times of k-means. 

For new data points, quantizing them to one of the classes is quick easy:
``` python
codes = hk.find_cluster(your_data_points, max_depth=-1)
```
If `your_data_points` is more than one data, i.e., two-dimensional data, it returns a list, each element is a tree node corresponding to the specific data point in `your_data_points`. For a single data point, it just return one tree node. `max_depth` tells the algorithm to find quantization at most in depth `max_depth`, default is `-1`, that is, quantizing the data to leaf nodes. If `max_depth` is given, the algorithm returns the quantization results(non-leaf nodes if `max_depth < ` depth of the branches) of the specific depth. 

To get the training data id of a specific leaf node(for example, finding k nearest neighbors), try:
``` python
leaf_node.get_additional_data()
```
where `leaf_node` can be the leaf node returned by `hk.find_cluster`. 

For a non-leaf node, you can gather all the data id(or data) in its branch to `output` by
``` python
output = []
non_leaf_node.gather_data_from_leaves(output, gather_additional_data=True)
```
If `gather_additional_data=False`, it gathers `data` instead of `additional_data`(for this tutorial, its the centroids of some classes).

Another usage for this code is creating [vocabulary tree](https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision) aka BoW model. You can try it yourself.
