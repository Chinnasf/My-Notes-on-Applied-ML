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

## First Part

Main notes were done in my notebook.

**REMARK**: one could also use the [Legendre-Galerkin Method](https://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1145-19.pdf). Garlekin Method has its roots on Finite Element Method.

## Second Part

### IDENTIFICATION, OBSERVATION, AND STABILITY ANALYSIS OF DIFFERENTIAL EQUATIONS

[Paper](https://www.sciencedirect.com/science/article/pii/0005109894900299): . Ljung and T. Glad 1994: On global identifiability for arbitrary model parameterization.

Type of problems:

* Observation: under-contrained
    * Not enough data
    * Multiple solutions

* Inference: well-defined
    * Exactly enough data
    * Only one solution

*Estimation: Over-contrained
    * Too much data
    *Maybe no solution: usually because of noise

**DEFINITIONS**

* Observability: the state can be inferred from the measurements: $x$. 
* Identifiability: the parameters can be inferred from the measurements: $\theta$.
    * Global identifiability.
    * Local identifiability.
* Persistent exciting signal.
* epistemic uncertainty
    * Epistemic uncertainty is a type of uncertainty that arises from a lack of knowledge or incomplete information about a particular system or process. It's often contrasted with aleatoric uncertainty, which arises from inherent randomness in the system.
* aleatoric uncertainty
    * overfitting happens when this is learnt    

PINN mentioned as the trying to identify parameters of unidentifiable / identifiable PDEs.

PINNs can solve "index problems". It seems that SINDy and PINNs are complementary.

So, it is IMPORTANT to be able to identify whether a problem is identifiable or not; or, under which conditions is it, etc. Why?

* **OBSERVATIONAL BIAS**: change the data to fit the underlying physics
* **LEARNING BIAS**: change the loss function to include physics biases
* **INDUCTIVE BIAS**: modify the training algorithm to force physics biases 
* **DISCREPANCY BIAS**: change the model to incorporate physics ((slide 15))

(Sparse Identification of Non-Linear Dynamics) SINDy with Fixed Cut-off Thresholding

**REFERENCES**
* Implicit function theorem: slide 9
    * [Identifiability of nonlinear systems with application to HIV/AIDS models](Identifiability of nonlinear systems with application to HIV/AIDS models) 

* [Discovery of nonlineardynamical systems using a Runge–Kutta inspireddictionary-based sparseregression approach](https://royalsocietypublishing.org/doi/epdf/10.1098/rspa.2021.0883): **QUESTION**: what is the complexity of RG-SINDy?

* [Physics-Informed SINDy (PhI-SINDy)](https://link.springer.com/article/10.1007/s11071-024-09652-2): Physics enhanced sparse identification of dynamical systems with discontinuous nonlinearities 

**QUESTION**: what is the algorithm complexity of the family of SINDy algorithms? Do you need a cluster? is this for big data?

* [Universal Differential Equations for Scientific Machine Learning](https://arxiv.org/abs/2001.04385)
    * A UDE is a forced stochastic delay partial differential equation; this analysis requires the implementation of the neural ODE + SINDy on a node. 


RELATED VOLVO PROJECT: slide 19. Identification of dynamical systems: for equation discovery. 

* [Physics-informed learning of governing equations from scarce data](https://www.nature.com/articles/s41467-021-26434-1): PINN + SINDy + PINN-SR ==> Optimization ADAM


## BAYESIAN IDENTIFICATION
 **LIBRO DE CRISTIAN** que me regalo :p (deep learning, Goodfellow, 2016)

 WHEN APPLYING THE MSE, it is the same as assuming that your noise is gaussian; otherwise, one cannot use the loss as MSE. I wonder if the volvo noise is Gaussian. 

 SINDy == Max A POSTERIORI WITH LAPALCE PRIOR!! So, instead of going to well-known algorithms right away, it is important to characterize the noise of the dataset. Then, if your noise is not Gaussian, one would have to redo the analyses (slide 23). 

So, SINDy is actually a Bayes Method. **This section is particularly useful to teach why it MSE es typically used**.

**BAYESIAN SYNDy**

* [Sparsifying priors for Bayesian uncertainty quantification in model discovery](https://royalsocietypublishing.org/doi/10.1098/rsos.211823): 

* [Response Estimation and System Identification of Dynamical Systems via Physics-Informed Neural Networks](https://arxiv.org/abs/2410.01340)

* IT IS FUNDAMENTAL TO UNDERSTAND THE METROPOLIS HASTING ALGORITHM (how does it compares to the rejection sampling algorithm?)
    * Create a plot similar to the one of slide 25 for the [electron beam simulator](https://github.com/Chinnasf/Physics/blob/master/FUSION-EP/Plasmas/TWO_STREAM_ELECTRON_INSTABILITY.ipynb).

There's an acute importance of the prior (which directly relates to the noise of the dataset) for identifiability of a problem. 

**OBSERVERS**

What is the difference between the filter and an observer? 

* Luenberger observer: **linear systems**
* KKL observer: **non-linear systems** : On the Existence of a Kazantzis-Kravaris/Luenberger Observer: [arXiv:0903.0297](https://arxiv.org/abs/0903.0297)
    * KKL Observer Synthesis for Nonlinear Systems via Physics-Informed Learning: [arXiv:2501.11655](https://arxiv.org/abs/2501.11655)





