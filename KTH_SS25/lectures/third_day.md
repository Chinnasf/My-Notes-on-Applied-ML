# TRAINING AND OPTIMIZATION
Emanuel Strom

**DATASETS**

* [ European Centre for Medium-Range Weather Forecasts](https://www.ecmwf.int/en/forecasts/datasets)
    * [Real-time forecast data](https://www.ecmwf.int/en/forecasts/datasets/search?f%5B0%5D=filter_by_type_%3AReal-time)
    * Possible to implement regression


**QUESTIONS**

*What makes high dimensionality? How many features does it becomes too many features? 


**CONTENT**

* Non-convex optimization; what does non-convex loss mean? 
* $\eta$ perturbation: such that the average loss has an extra term with a perturbation (slide 7): $L(\theta - \eta\delta L(\theta))$
    * Issue, when gradient is zero: when there is a singular point.
    * Goal: converge to a singular point: minima or maximum 
* Gradient descent
    * Gradient descent and its convenience for high dimensionality dataset: investigate
    * PDE: forward euler == gradient descent. 

3 MAIN IMPROVEMENTS

1. GD with coordinate change    
* Scaling all the way with B (slide 10); this is why it's not the same as changing the initial guess.
    * Geometry depiction of the improvement fr gradient descent with change of coordinates. Finding the minima faster
     
2. Memory --> Acceleration
* This gives GD memory; so it knows that if it has come from a local minima... 
* Slowly foret information over iterations
* Incorpate future prediction: Nesterov trick (slide 11)
* Even these algorithms struggle to get out of local minima

3. Stochasticity
* Adding a noise term with zero expectation
* Adding a noise is not that you are getting a distribution function, like Gaussian, and then applying GD
    * Adding a noise in this slide is that one is doing the GD with batches; the fact that you are randomly taking subsets of your dataset such that
    * All of the subsets have the same expecation value; this term W_k is the mathematical recognition that by doing gradient descent with batches, adds noise to the analysis. 
    * When you pick the batch, do it uniformly; sample uniformly. You want that each expectation does not deviate from the others, otherwise, the optimization is not for the whole underlying distribtion function.
* Overdampend Langevin dynamics: relation to Brownian motion; again, Physics opening the way to ML.
    * Partition function compared to Stochastic gradient descent
    * Eta, stepping value.
    * Noise is W_k
* Ask for a comprehensive introduction to Stochastic GD. :v  

*TASK: create a comparison between GD and  stochastic GD 


*IMPORTANT: ADAM: coordinate wise reescaler; with stochasticiy and momentum. Read the paper. 

* **A discussion on deep learning**: The mathematical and theoretical work on no bad local minima for deep NN.

**PRACTICAL TIPS**

* Field: statistical learning theory
* Explain why ReLU is a big big no in DeepNN.
* Work with unit-independent data so that the learning rate also does not depend on units; for instance, optimizing th speed over a geometry wing. If not notmalized, then gradient may also be taking values of speed, when there are other dependent variables in the dataset. 

**VANISHING and EXPLOIDING GRADIENTS**

* Information compression: `xavier_normal` with torch. SLIDE 18. But many not ideal for PINNs. Bottle-necking: compressors. 

* BATCH NORMALIZATION LAYER: adding a virtual layer in between the layers: helps with making all the gradients equal in magneidude. 

**Question**:  how does PCA compares to data compression/ bottlenecks? Can it be used as a data preprocessing step? Models for preprocessing? 
    * PCA then compression; this has been implemented, I believe.

* The choosing of the activation function matters because it can filter out, let's say, already exploiding datasets. I gues, for instance, one of them is tanh(x).

* ELEMENT WISE MULTIPLICATION vs TENSOR PRODUCT? In the paper, they use the same symbol, but they are NOT the same. 

Okay, so, optimization algorithms are designed precisely so that exploiding or vanishing is not observed. Mention and classify the different optimizers based on the three types of optimizarions. 
z.B.:

* AdaGrad 
* Nadam: nesterov adaptive multiplication 
* Lion Optimizer
* Rprop Optimizer 
* Quasi-Newton Methods: so that the inverse of the Hessian is not hell (MIND YOU! Not stochastic by nature); however, one can warmstart with a stochastic method, then implement this. 
     * Broyden-Fletcher-Goldfarb-Shannon Method (BFGS Method): optimizing in a way that there's memory with small corrections. 
        * Rank-two correction to the matrix: Anzats in slide 39: $H_{n+1}$
            * This is because one-rank correction cannot promise positive definite; otherwise, if you need two conditions to fulfill, you need two dregrees of freedom.
            * To understand the previous, ask the question: why is it important a two-rank correction in the anzats for BFGS method and not one-rank correction is sufficient?
     * Limited-Memory Broyden-Fletscher-Goldfarb-Shannon Method 
     * Mixed implementation, z.B.: NN + Function Approximation + ADAM + L-BFGS
* See slide 44. 
    * Related: **REFERENCE**: [Which Optimizer Works Best for Physics-Informed Neural Networks and Kolmogorov-Arnold Networks?](https://arxiv.org/abs/2501.16371)
    * Subsequent slides show the exploration of different learning rates. 
    * Related: **REFERENCE**: [Snapshot Ensembles: Train 1, get M for free](https://arxiv.org/abs/1704.00109)

IT IS NOT STRAIGHTFORWARD WHICH OPTIMIZER IS GOING TO WORK BEST FOR YOU: code in a way that it is possible to test different optimizers. Do not speed too little time with them, also. Make sure you understand well why which optimizer may not be the best for your and then explore other options. 
One could also start with something stable and then explore the learnng parameters to explode them and make them fail. Really explore the extremes: too large and too little. (slide 33 of the second part).

**REFERENCES**
* EVERYONE IS READING THIS: [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
* [Paper Summary: Understanding the difficulty of training deep feedforward neural networks](https://karan3-zoh.medium.com/paper-summary-understanding-the-difficulty-of-training-deep-feedforward-neural-networks-6f184ba5e6d6)

**REMAKR**: fixing the last layer is a quadratic problem, why? Is this only for fully connected NNs?: THIS IS REGARDLESS IF THIS IS A FULLY CONNECTED OR NOT NN. 
* Linear PINN? For instance, for a Poisson equation. slide 59.

Him: "for me, Resnet always perform better (PDE context)". 

# FILTERING: The Basics: [Jennifer Ryan](https://intra.kth.se/sci/skolinformation/interview-with-jennifer-ryan-1.1179008)

MAKE SURE YOU USE THE RIGHT FILTER FOR THE RIGHT APPLICATION.

* References: notice how most of them are from the 70s and 80s: "there has not been much progress on filters". 
    * Definition
    * important
    * Statistical relation 

Motivation: [Nodal Discontinuous Galerkin Methods](https://link.springer.com/book/10.1007/978-0-387-72067-8) -- you have the book. 

* Reduce Gibbs and Recover accuracy: filter such that the shock is resolved. However, the role of filter is not to recover information of a shock, but what's recorded before it. This is why this lecture. 

**Important***: truncated Fourier series. But before we dwelve into this, MAKE SURE YOU UNDERSTAND what are the assumptions in order to apply Fourier Series; z.B. decaying function, continuous and periodic? etc.
    * This opens the discussion: decay of Fourier parameters: comparison to vanishing parameters in GD. 
    * If we are interested in studying the jump in the discontinuity, then... THERE IS A SLIDE with an integral that says "sin(s)/s", I think. She says: "this is never shown and this is what is introducing your oscillations".


