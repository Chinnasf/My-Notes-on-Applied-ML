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
     * [On the connection between the conjugate gradient method and quasi-Newton methods on quadratic problems](https://arxiv.org/abs/1407.1268)
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

**REGULARIZATION**: overfitting vs underfitting
Slides show how to implement in tensorflow. 

* A comparison between $L_1$ and $L_2$ is mentiond. Dropout in NN is quite similar to $L_2$ regularization... is it tho? Shoudn't it be $L_1$? Because $L_1$ can be used for feature selection and there are weights that can be set to zero. 
* REMARK: usully, apply dropout only to the higher 1-3 layers but not to the output layer. 
* IMPORTANT: dropour is active during training, evaluate NN without dropout after training. 

SEMINAR: [https://www.youtube.com/watch?v=XL07WEc2TRI](https://www.youtube.com/watch?v=XL07WEc2TRI)

THIS IS AAAAAALLLLLLL ADVANCES STATISTICAL PHYSICS:
* Mutual Information 
* KL-analysis
* Data Processing Inequality and Invariance
* Encoders and decoders

Slide 85 shows that a 100 DNN compress the data, this is by studying the amount of information of input and the amount of information for output. 
* Apparently, this is not an intended consequence, it was someone whose mind was focused on studying the behaviour in terms of indormation. Some people care about the parameter space, the architecture of the NN, nut this person was focused on the input/output information. 

* [REFERENCE](https://arxiv.org/pdf/1503.02406): Deep Learning and the Information Bottleneck Principle
* [REFERENCE](https://ieeexplore.ieee.org/document/7133169): Deep learning and the information bottleneck principle -- Imperial college longon has access to this. 






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
    * Global decay rates vs ... 

** HOW TO APPLY FILTERS?**
* Modify the fourier coeffs in order to observe the decay in a smother way. Decrease the oscillations. Remark: the modification is in a way such that it affects the "pollution" of the discontinuity. 
* This is were PINNs can be used: creating PINNs such that they behave as filters. 
    * A filter $\sigma(\eta)$: what is a filter? Mathematical definition, slide with reference "H. Vandeven, 'Family of spectral filters...' year 1991". 
* What is the connection between frequency filters and wavelenght filters?
* Most of the information is on thhe low modes, why? However, we want to make sure they're unaffected. 
    * Why not just cut the hight modes? Simplest answer: loss of information. 

* REMARK: modified Fourier space (signal space) implies a convolution in the physical space. ---> DIRICHLET KERNEL. 

* Fourier space = signal space
*  Euclidean space = physical space

* Spectral approximation: $e^{i\omega_n x}$; thus, one needs to project the function onto space representation by {$e^{i\omega_n x}$}^{N/2}_{n=-N/2}.
    * Projecting onto basis vectors is like projecting onto approximation spcace. 

* THIS IS A STATE ABOUT plasma, IN A SLIDE FROM SOMEONE IN LOS ALAMOS:
* SIAC filters reduce the number of time steps required to 2600 time-steps 1.3% of data: THE SMALL SCALE NOISE DOES AFFECT THE LONG-TERM CONPUTATIONS. 
    * SMoothness-Increasing-Accuray Convervinf (SIAC) Filter
    * [Resource](https://www.sciencedirect.com/science/article/pii/S0045793020302978): Enhancing accuracy with a convolution filter: What works and why!

**WORKING STEPS**
1. Choose the numerical approximation method to represent the fiven data
2. Choose the filter that matters for you.
3. Apply a filter
    * Applying the filter to try to optimize the coefficient for the given expression. Have a look at the photo. 
    * The filter is a convolution kernel function in the physical space when applied in the Fourier space. 

Now, we don't want to loose information. So the way in which we represent weights matter. And, the values that these weights are chosen are dependent on the way in which it has been chosen how the function or data will be modelled. 
* How does one choses $r$? $r$ is chosen for accuracy. 

"In Fourier space, is where dissipation and dispersion matters". 

"If I use a polynomion of order two, I expect third-order accuracy."

"There's higher order information in my frequency space"

"Filters are just PDFs"

There are constrains that must be complied in order to not loose information (moments and  consistency). Once this is confirmed, one needs to look at the noise of the data. Determine which information you can get from data, 
For instance, she is showing an analysis on the pattern of oscillations. "We can understand noise, for how numerical approximations are contructed." 

The term that one must concentrate is the one that arises when studying the error. 

Projection always allows to retreive a higher order of indormation than interpolation. In which, projection requires the choosing of a kernel; the kernel can be linear, quadratic, etc. However, you cannot ignore the required properties in physical space nor frequency space.
Have a look at the "general framework: weights": enforce consistency and moments. The slide that looks like quantum mechanics. 




**QUESTION:::: DOES THE AMOUNT OF DECIMALS MATTER FOR YOUR FIELD? Is it because of a education purpose or is it because the application requires that amount of decimals**.
* GLOBAL FILTER: you're using all the data. What does global filter mean? Is it a downside? 
* How does one handle high frequency oscillations?

**KEYWORDS**
* Dissipation 
* decay rate*smoothness / accuracy 
* negative values ?
* Spectral approximation.
* Spectral methods.
* Amplitude matrices.
* **Negative-order norm estimates**: gives information about the noise. --> higher order information. This gives more information than MSE and tells how to extract the information from the data. 

"Filtering is just not to reduce about Gibbs, but also to save energy and time (computational power)" as in the long run, these accumulate and matters. Her work is about optimization on the filtering also close to shock registration and rapid frequency signals.

Mean and variance are statistical quantities that one is ALWAYS interested to present. This, hence, affects the way in which filters are designedl such that physical space behaviour is complied. 

Filer = Kernel 

OBJECTIVE: construct convolutional filters based on a linear combination of function that translates and obtains its corresponding PDF. 

This person is the invertor of this method, along with [Andres Galindo-Olarte](https://www.linkedin.com/in/andr%C3%A9s-felipe-galindo-olarte-9434b4188/) (Colombia), from University of Texas-Austin. 


"ALREADY HAS DEMONSTRATED IT'S ABILITY TO REDUCE THE AMOUNT OF DATA NEEDED IN PIC PLASMA SIMULATIONS". sile 15

CONCLUSION:

* "I have described a general framework, choose how discipative you want to be, etc" --> how to design your filter. 
    * [Denoising Particle-In-Cell data via Smoothness-Increasing Accuracy-Conserving filters with application to Bohm speed computation](https://doi.org/10.1016/j.jcp.2024.112790)
        * THIS READING IS A MUUUUUUUUUUUUUSSSSSTTTTTTTTTT
        * Highlights
            * SIAC filters are effective denoisers for moment data arising from PIC simulations.
            * SIAC captures appropriate Fourier signal information.
            * SIAC reduces amount of information necessary in computation of quantities of interest.
            * SIAC is useful in Bohm speed calculation.  
    * [Here is the code](https://gitlab.com/msiac-tool/MSIAC) in Julia
    * [Here is the website](https://siac-magic.gitlab.io/web/)
    * Some of the published work explains more or less the the framework presented here, but, apparently, there's no publication work were the content of this presentation is in a written format; however, the References rection show the connection of all of these.    

* UNDERSTAND how your filter must behave --> understand convolution
    
During time integration analysis:
* are we still conservative?
* entropy solution? --> physically relevant

Picklo & Edoh: Journal of Scientific Computing --> requires the modification of the filter. 
* THEIR WORK: [Entropy Correction with SIAC Filters for High-Order DG Methods](https://link.springer.com/article/10.1007/s10915-025-02905-1)


* Photo with email. 


# IMPORTANT FOR GENT PHD

* Make sure you understand the modeling/simulation (?) processing of signal with shocks, as many probes may pick that from tokamaks. 
* https://iopscience.iop.org/article/10.3847/2041-8213/ab398b: 

* Took a photo for inspiration: "this is where the methods make a difference, this is the application of PINNs". However, notice how there's also a noticeable difference in the analytic function for the decay rates. 
* Her: THE "DIRICHECK KERNEL IS THE MOST DISSIPATIVE: |sin(2x)|": going away from the actual form. For the heaviside function, none of them are even getting close to the discontinuity. However, some of them, help to surpress the Gibbs oscillations. 
    * We have  "another filter", however, it does not satisfy the definition of a filter! **the exponential filtar**. Alpha is the strength of the filter, and eta is the order of the filter, with a cut-off.
    * Order of the filter controls accuracy, alpha --> dissipation (how fast you're cutting off), eta-c --> frequency cut-off mode; this has been applied to "Fourier data" 
    * When one does not want to sacrify information. 
    * Remark: kernel becomes negative.
    * What is the scaling? 

### CURIOSITY QUESTION:

* Can I create a PINN for modelling the POISSON equation for the two stream instability problem? 
* Will it be interesting for Gert that I model the whistler instability? 

