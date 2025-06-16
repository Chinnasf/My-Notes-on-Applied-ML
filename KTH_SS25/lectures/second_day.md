# Introduction to Neural Networks and PINNs

* [Data structure](https://www.h-schmidt.net/FloatConverter/IEEE754.html)

Important: there are 14 different types of ML algorithms. see slike ~ 58.

## Where to get datasets

* [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/)
    * [Challenger USA Space Shuttle O-Ring](https://archive.ics.uci.edu/dataset/92/challenger+usa+space+shuttle+o+ring)

### IMPORTANT CONCEPTS

* Single layer as a Universal Functional Approximation.
* Hidden layers
* Activation functions

* INFORMATION THEORY (slide 38)
    * Entropy vs Cross-Entrpy
    * KL Divergence: non-symmetric distance metric
    * KL-distribution divergence

Understanding the entropy of a dataset directly relates to the degree of computability. Essentially, it signifies a level of computational difficulty where the problem is not just unsolvable by a standard computer, but also its solution is as complex as it can be while still being computable from the halting problem. 


Workflow of a neural network.

1. Input Layer.
2. Hidden Layers: defines the depth of the NN. 
    * Width of the NN is the number of neuron in the hidden layer. 
3. Output Layer.


Steps for optimization in NNs

* Forward pass
* Automatic-differentiation
* Backward pass

All three of them are available in Tensorflow, Pytorch, and Jax. 

### NN for regression and Classification

Last hidden layer in regression progrems must be linear (why?). For classification, the _softmax function_ must be defined. These are under the assumption to fully connected NNs.

### ResNet: residual NN

Here, some connections are skipped. The benefit is that learning is facilitated, as this reduces the complexity and hyperparameters to be learnt. 

## Data + Physical Laws 

This field dircerns between three scenarios:
* Small data, lots of physics.
* Data and physics available.
* Only data and no physics available (disruptions)

So, it seems that PINNs are a form of UNSUPERVISED LEARNING. It requires collocation points; this, does not necessarily mean that they depend on grids, but the idea is comparable. 

* Example of [heat transfer](https://www.arxiv.org/abs/2502.00552): a comparison between simulation and PINNS. There is a sensor placement. 
    * Here, one knows the parameters, the objective was to find a solution in a faster manner. Accuracy was not as good as simulations, but it was certaily faster.
    * For instance, this was more or less comparable to the objective of DC15; where one would take prefer a faster and more-or-less accurate parameter estimation than simulation so that the bridging multiscale problems.

* Example of [insulation materials](https://ieeexplore.ieee.org/document/10043884): a power transformer's lifetime is directly associated to its insulation. On average, it is 40 to 60 years as a lifetime.
    * Here, the paper focuses on **the discovery of unknown parameters**. $A$ and $E$ were the target parameters so that the aging process of insulators could be known better.
    * **Could one use symbolic regression to get to understand disruptions in tokamaks?**
        * [2016 research paper](https://iopscience.iop.org/article/10.1088/0029-5515/56/2/026005): Application of symbolic regression to the derivation of scaling laws for tokamak energy confinement time in terms of dimensionless quantities.
        * [2020 research paper](https://www.mdpi.com/2076-3417/10/19/6683): Investigating the Physics of Tokamak Global Stability with Interpretable Machine Learning Tools
        * [2025 research paper](https://www.nature.com/articles/s42005-025-02023-2): Discovering nuclear models from symbolic machine learning
        * [slides](https://nucleus.iaea.org/sites/fusionportal/Pages/DPWS-6/TM%20Fusion%20Data%20Processing%20Validation%20and%20Analysis/3_Wednesday/Session%20III%20Regression%20Analysis%20Profiles%2C%20Scaling%20and%20Surrogate%20Models%20(Cont)/840%20Murari%20A.pdf): Data Driven Theory: how to derive mathematical models directly from data.
        * [2024 research paper](https://arxiv.org/abs/2404.11477): Discovering Nuclear Models from Symbolic Machine Learning
 
    * Library in Python for Symbolic Regression: [PySINDy](https://pysindy.readthedocs.io/en/latest/).

 

* Example of [Discovering Partially Known Ordinary Differential Equations: a Case Study on the Chemical Kinetics of Cellulose Degradation](https://kth.diva-portal.org/smash/get/diva2:1955083/FULLTEXT01.pdf)
    * "Symbolic regression in order to convert it into an expression"
    * May not work on stochastic estimations; however, this has not been tried. 

### REFERENCES

[On hyperparameter optimization of machine learning algorithms: Theory and practice](https://www.sciencedirect.com/science/article/pii/S0925231220311693)


# INTRODUCTION TO DIFFERENTIAL EQUATIONS

