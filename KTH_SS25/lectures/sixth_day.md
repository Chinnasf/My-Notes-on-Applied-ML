# Scientific Computing Enhanced by Model Order Reduction and Machine Learning, Gianluigi Rozza

Particularly important for surrogates. 

Numerical analysis: not so famous but really important. 

[SIISA](https://mathlab.sissa.it/) has an open CFD source (mathlab) tho. zB.: [rbnics](https://www.rbnicsproject.org/). 


# UNCERTAINTY QUANTIFICATION

Any measurement without noise is useless. 

## Uncertainty Quantification in Scientific Machine Learning: Methods, Metrics, and Comparisons [arXiv:2201.07766](https://arxiv.org/abs/2201.07766)

This is like a book but it covers everything. 

There are 8 major sources of uncertainty. 

* Aleatoric vs Epistemic uncertainty. 
    * Epistemic uncertainty cannot be reduced. 

FUNDAMENTAL: understand the difference between frequentist and bayesian approaches. 

It is possible in both approaches, frequentist and Bayesian, to estimate the uncertainity. 

### Bayesian Methods

Explanation: Bayes Theorem. 

* There are two terms: veryfication and validation. 
    * Sensitivity vs Specificity (these cannot be studied with the frequentist approach).
        *   **TASK**: use your Bayes explanation on femicides to also explain these terms. 

### Bayesian Model Average (BMA)

It's possible to estimate the uncertainty by considering a bunch of $\theta$. 

**TASKT**: Explain why Hamiltonian Monte Carlo (HMC)​ is needed in this context. 
* Key: statistical ensemble vs canonical state

OPTIMIZED VERSION OF THE HMC ---> No-U-Turn... 

CHEAP VERSION OF THE HMC ---> Langevin dynamics :: a solution to the stochastic OD


# VARIATIONAL INDERENCE

Supervised learning. 

Today, NNs can have trillion parameters to be optimized; Bayes methods require the intregation of the parameter space. For that, there are different sampling methods: MCMC, Langeving dynamics, etc. 

Here's how people approach this: ensembles (like random forests). 

Ensemble combine multiple neural networks to produce uncertainty estimates by capturing variations across model weights. But, why can we think of ensambles as Bayesian inference? 

Bayesian models average assumes that one paameter setting is correct, and averages over models due to an uncertainty ... HAVE A LOOK AT THE SLIDES.

## MOST IMPORTANT RESOURCES 

* [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://proceedings.mlr.press/v48/gal16.html)
    * [https://proceedings.mlr.press/](https://proceedings.mlr.press/)

* [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142)