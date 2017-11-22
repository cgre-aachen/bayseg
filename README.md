# BaySeg

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)]()

A Python library for unsupervised clustering of n-dimensional datasets, designed for the segmentation of one-, two- 
and three-dimensional data in the field of geological modeling and geophysics. The library is based on the algorithm 
developed by [Wang et al., 2017](https://link.springer.com/article/10.1007/s11004-016-9663-9) and combines Hidden Markov
Random Fields with Gaussian Mixture Models in a Bayesian inference framework. It currently supports one physical 
dimension and is in an early development stage, but we are working tirelessly on increasing its efficiency, ease of use
and expanding the implementation to two and three physical dimensions.
 
### Examples
#### Segmentation of geophysical well log data
![alt text](data/images/front_gif.gif)

(Above well log data used from machine learning contest of [Hall, 2016](https://library.seg.org/doi/abs/10.1190/tle35100906.1))

### Installation

As the library is still in early development, the current way to install it is to clone this repository
and then import it manually to your projects. We plan to provide convenient installation using PyPi in the future.

#### Dependencies

BaySeg depends on several genius components of the Python eco-system:

* `numpy` for efficient numerical implementation
* `scikit-learn` for mixture models
* `scipy` for its statistical functionality
* `matplotlib` for plotting
* `tqdm` provides convenient progress meters

#### Cloning directly from GitHub

First clone the repository using the command (or by manually downloading the zip file from the GitHub page)

    git clone https://github.com/cgre-aachen/bayseg.git

then append the path to the repository:
    
    import sys
    sys.path.append("path/to/cloned/repository/bayseg")
    
to import the module:

    import bayseg

### Getting Started

Instantiate the classifier with the physical coordinates vector, the feature vectors and the number of labels:

    clf = bayseg.BaySeg(coordinates_vector, feature_vectors, n_labels)
    
Then use the _fit()_ method to classify your data with your desired number of iterations:

    clf.fit(n_iterations)

### References

* Wang, H., Wellmann, J. F., Li, Z., Wang, X., & Liang, R. Y. (2017). A Segmentation Approach for Stochastic Geological Modeling Using Hidden Markov Random Fields. Mathematical Geosciences, 49(2), 145-177.
* Hall, B. (2016). Facies classification using machine learning. The Leading Edge, 35(10), 906-909.