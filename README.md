# BaySeg

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)]()

A Python library for unsupervised clustering of n-dimensional data based on a combination of Hidden Markov Random Fields,
Gaussian Mixture Models and Bayesian inference. The algorithm is based on the work of
 [Wang et al., 2017](https://link.springer.com/article/10.1007/s11004-016-9663-9). It currently supports 1-dimensional
 physical space and is in an early development stage.
 
![alt text](data/images/front_gif.gif)

(Above well log data used from machine learning contest of [Hall, 2016](https://library.seg.org/doi/abs/10.1190/tle35100906.1))

### Installation

As the library is still in early development, the only way to install it is to clone this repository
and import it manually to your code:

Clone the repository:

    git clone https://github.com/cgre-aachen/bayseg.git

Append the path:
    
    import sys
    sys.path.append("path/to/cloned/repository/bayseg")
    
Import the module:

    import bayseg

### Getting Started

Instantiate the classifier with the physical coordinates vector, the feature vectors and the number of labels:

    clf = bayseg.BaySeg(coordinates_vector, feature_vectors, n_labels)
    
Then use the _fit()_ method to classify your data with your desired number of iterations:

    clf.fit(n_iterations)

### References

* Wang, H., Wellmann, J. F., Li, Z., Wang, X., & Liang, R. Y. (2017). A Segmentation Approach for Stochastic Geological Modeling Using Hidden Markov Random Fields. Mathematical Geosciences, 49(2), 145-177.
* Hall, B. (2016). Facies classification using machine learning. The Leading Edge, 35(10), 906-909.