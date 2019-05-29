# BaySeg

> Easy-to-use unsupervised spatial segmentation in Python.

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)]()
[![Python 3.6.x](https://img.shields.io/badge/Python-3.6.x-blue.svg)]()
[![Build Status](https://travis-ci.org/cgre-aachen/bayseg.svg?branch=master)](https://travis-ci.org/cgre-aachen/bayseg)

## Contents

+ [Introduction](#introduction)
+ [Examples](#examples)
  - [1D: Segmentation of geophysical well log data](#1d-segmentation-of-geophysical-well-log-data)
  - [2D: Combined segmentation of geophysical and remote sensing data](#2d-combined-segmentation-of-geophysical-and-remote-sensing-data)
+ [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Cloning directly from GitHub](#cloning-directly-from-github)
+ [Getting Started](#getting-started)
+ [References](#references)
+ [Contact](#contact)

## Introduction

A Python library for unsupervised clustering of n-dimensional datasets, designed for the segmentation of one-, two- 
and three-dimensional data in the field of geological modeling and geophysics. The library is based on the algorithm 
developed by [Wang et al., 2017](https://link.springer.com/article/10.1007/s11004-016-9663-9) and combines Hidden Markov
Random Fields with Gaussian Mixture Models in a Bayesian inference framework. It currently supports up to two physical 
dimension and is in an early development stage.
 
## Examples



### 1D: Segmentation of geophysical well log data

![alt text](data/figures/front_gif.gif)

(Above well log data used from machine learning contest of [Hall, 2016](https://library.seg.org/doi/abs/10.1190/tle35100906.1))

### 2D: Combined segmentation of geophysical and remote sensing data

You can try out how BaySeg segments 2D data sets by using an interactive Jupyter Notebook in your own web browser, enabled by Binder:

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/cgre-aachen/bayseg/master?filepath=notebooks%2Ftr32_presentation_example.ipynb)


## Installation

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

## Getting Started

Instantiate the classifier with the n-dimensional array storing the data and the number of labels:

    clf = bayseg.BaySeg(data_ndarray, n_labels)
    
Then use the _fit()_ method to classify your data with your desired number of iterations:

    clf.fit(n_iter)

## References

* Wang, H., Wellmann, J. F., Li, Z., Wang, X., & Liang, R. Y. (2017). A Segmentation Approach for Stochastic Geological Modeling Using Hidden Markov Random Fields. Mathematical Geosciences, 49(2), 145-177.
* Hall, B. (2016). Facies classification using machine learning. The Leading Edge, 35(10), 906-909.

## Contact

The library is based on research [Hui Wang](https://www.researchgate.net/profile/Hui_Wang122) and [Florian Wellmann](http://www.cgre.rwth-aachen.de/go/id/qpan/lidx/1/gguid/0x5440F5A53D654C41874F09C577FE4005) for a research project in the German Collaborative Research Center [SFB TR32](http://www.tr32db.uni-koeln.de/site/index.php). It was rewritten in Python from a Matlab code by [Alexander Schaaf](https://www.researchgate.net/profile/Alexander_Schaaf4).

Bayseg is currently being developed by the LuF Computational Geoscience and Reservoir 
Engineering (CGRE) and the Aachen Institute for Advanced Study in Computational Engineering Science (AICES) at RWTH Aachen University, Germany.

For more information and contacts, please see: http://www.cgre.rwth-aachen.de/


