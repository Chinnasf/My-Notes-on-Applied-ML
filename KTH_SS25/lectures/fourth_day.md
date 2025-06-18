#  Neural Network Architectures

**REMARK**: Make sure you understand what are the properties of the last hidden layer. 

* Weierstrass theorem
     * [Best Approximation](https://www.sciencedirect.com/topics/mathematics/weierstrass-theorem#:~:text=Weierstrass%20Theorem%E2%80%94Existence%20of%20a,a%20closed%20and%20bounded%20set.)

DEPENDING ON THE DATA YOU CHOOSE THE ARCHITECTURE

### QUESTIONS
* Are CNN only used for images?
* So, LSTM comes as a solution for vanishing and exploiding gradients of Recurrent NN?


### REMARKS and KEYWORDS

* PDE with no dissipation or dispersion makes it the most difficult to solve, why? all numerical methods have that. One example of it is wave equation. 

* Three sections of learning:
    1. Fitting
    2. ...
    3. ...

* Fitting is chosen on the basis of the parameters of the data

* MAKE SURE YOU UNDERSTAND: Bottleneck approuch w.r.t. dimensionality reduction

* Spectral bias in NNs

* NEURAL NETWORKS are universal approximations
    * REFERENCE: [Universality of deep convolutional neural networks](https://www.sciencedirect.com/science/article/pii/S1063520318302045)
    * There's a theorem

* CNN also behaves as an operator. 

* Numerical speed should be always less than the physical speed, when working with simulations. 

* "No body expects you to converg in a NN, but you have to converge to the best local minima"

* DO NOT FORGET TO STUDY THE STABILITY, CONVERGENCE, AND ... of the neural network: global and local sense. 
    * Numerical error vs approximation error
    * Numerical dissipation vs physical dissipation

### CONVOLUTIONAL NN and FNN vs CNN

Convolutional Neural Network works well for images recognition. 

Workflow: (see slides to complete)
* Input image: 28x28
    * get features: 6 structures 


ReLU is good for the image processing community, but really bad for the scientific community; "we will see why". 

Convolution is about filtering. 

How do you take the gradient of an image? Taking the difference between slices within the image. This action is called convolution. 
* Because this is the gradient, convolution enhances the features; it also reduces the size of input. 
    * WHEN YOU DO THE CONVOLUTION, MENTION WHICH DIRECTION YOU ARE PERFORMING THE DIFFERENCE. 
        * Some people put a comment on the output size given the input size. 

* The need of parametrization is to achieve learning and generalization, sometimes we do not want to reduce the number of parameters; this could imply the limitation of the learning capabilities. 
    * In some cases, you do not want your dimensionality to be reduces. 
    * By applying convolutions, one looses pixels, this is why **padding** is then performed. 
    * **stride**: when is it needed?
        * CNN used in turbulence. I think this is one type of requirement for this field, I believe not everyone needs this.  

Graph NN are born from CNN; so, understanding CNN can open the doors to having a better understanding to other architectures. 
* "The one issue with Graph-CNN is that it does not scale; but it is a good tool for shape optimization. It did not scale well.".  


### RESIDUAL NEURAL NETWORK

The deeper the NN, one can observe that the layers 'far away' from the input, start to "forget information"; so the idea of this architecture is to add the input to the outpur. 
* [REFERENCE](https://link.springer.com/article/10.1007/s40304-017-0103-z): A Proposal on Machine Learning via Dynamical Systems

* How to visualize a loss function? One may need to perform PCA as it may depend on high dimentionality.


### RECURRENT NEURAL NETWORK

* Sequential data
    * This requires a special preparation before putting it into RNN

* Common issues: vanishing and exploding gradients. 


### LSTM: LONG-SHORT TERM MEMORY NEURAL NETWORKS

* Time-dependent systems benefit from this architecture.

### ENCODER-DECODER ARCHITECTURE

* One cannot learn a delta distribution with regular NN. However, it is possible with encoders. This is where the bottle-neck description comes into place. 
* This helps to reduce dimensionality 
* Are encoders only for images? I would guess not. 
* Is there information loss with the compression? Reduction of dimensionality. Feature enhancement --> .

* How does one studies the information loss when working with compression? 
    * Single value decomposition vs PCA: there's also a loss of information, no one notices that. 
    * Dissipation == loss of information

* PCA: a manifold on which data lies. 

ENCODER DECODER ARCHITECTURE: how does the output changes when one performs PCA before that input w.r.t. the case you did not apply PCA. 

* WHEN YOU HAVE A LAPLACIAN, THAT IS DIPISSATIVE. 
    * You need to know from where your data is comming. Is your data dissipative? 
        * If yes, then this directly impacts how much information you are loosing and how much can be recovered when using decoders.    
    * Paper? 
    * Trace of a matrix required. 

### EMBEDDINGS

* What is an embedding? 
* Positional embedding: z.B.: what's the importance of a word w.r.t. others in a sentence. Highly required by GPT. 
* Attention Mechanism

* Photo for how softmax was adaptep for a PINN towards PDE.


### GENERATIVE MODELLING

* Goodfellow et al 2014
* They have a very specific loss function 
    * Jensen-Shannon divergence 
* Diffusion Models --> Unsupervised learning. 


#### HOW DO YOU NORMALIZE YOUR DATA

NNs do not like not normalized data

* PINNS: minmax works well with minmax. See photo. 

x --> \frac{ x - \mu }{ std(x) }


# Discovering Differential Equations: INTRO

* C++ --> pYTHON
    * Derivation of a python code? 

* Dimensionality of the dataset matters to which mode you are using: forward or reverse mode. 
* Reverse mode is everywhere. 
* Forward mode is better for the wave equation, then? 

JAX --> a differentiable numpy: jax-numpy `jnp`. 

### JVP vs VJP (forward-mode autodiff)

Jacobian-Vector Product(JVP) or Vector-Jacobian Product(VJP). 

Micrograd. 

Differentiable programming. 

Gradient of losses are more pronounce than the pinns. 


# Discovering Differential Equations

Dynamical system: why do we care? Control problem. 

### If you have data, how do you discover the equation?

Suppose you have a vector, and this vector is divergence free. Like the continuity eq in navier stockes equation. 

Define a NN such that it follows the physical system (like, conservation of energy, etc). You will have soft constrain (modified loss function) and hard constrains (NN must follow that structure). 

* A numerical solver can be used for lots of physics with small data.

In the following lectures

* Some Physics and some data.
* Social dynamics (no physics at all, like social dynamics).


Why data and physics is important? 

* SYMBOLIC REGRESSION

For conserving properties, like in the hamiltoninan, one must use specific integrators. Not everything will be suitable. 

READ:
* [Hamiltonian Neural Network](https://arxiv.org/pdf/1906.01563)
* [Lagrangian Neural Network](https://arxiv.org/pdf/2003.04630)

### What is the difference between a ResNet and a ODENet?

* Canonical neural network 

* Replacement of forward pass with an ODE solver ( any ODE solver (idea 1964) there's a theorem -- BUT I GUESS THAT THE PROPERTIES OF THE SOLVER MATTER, NO? JUST LIKE SOME SOLVERS WORK BETTER FOR A CONSERVATIVE HAMILTONIAN).

* RESNET: Number of layers relate to the time step in the solver

Example of regression for a 1D function (jax jpnb tutorial)

Heart of code: 

CANONICAL APPROACH OF THE NN VS ODENet

* Canonica: W_{n+1} (board)
* ODEnet: forward pass replaced with ode-iintegrator
    * "Don't use any ODE integrator, there's `jax.experimental.ode`; this is more efficient"
    * Same routine with gradient descent 

t=0 --> Input of a neural network 

### ADJOINT METHOD

A linear function + a constrain example.

HOW YOU SOLVE EVEN THE MULTIPLICATION OF OBJECTS AFFECT THE PERFORMANCE OF THE CODE (see slide 20).

Adjoint methods are exploided when the chain rule of derivatives may be required. One may encounter: the product of the adjoint with the jacobian. 
* See the proof. 
* Modify the ODE solver to recover properties; also, it helps to reduce the complexity of the model (photo). 

* CAN YOU APPLY THIS TO A LORENTZ SYSTEM? (it's in the jpnb)
    * Or to a damped harmonic oscillator?

* `torchdiffeq` --> INSTALLED IN `aGPUo`.

### MULTI-STEP NN. 

* [Adams-Bashforth and Adams-Moulton methods](https://en.wikiversity.org/wiki/Adams-Bashforth_and_Adams-Moulton_methods)

IT IS ALL ABOUT HOW YOU INTEGRATE THE ODE. 

### HAMILTONIAN SYSTEMS: SympNets

Mi carnal agrego el three body problem en los codigos. 

There is a fucking poisson NN! How many are there out thereeeeeeee. Amazing. 

There's a fucking non-linear schrodinger equation nn. I'm speechless. 


# INTRODUCTION TO PINNs: PART ONE

Suppose you want to simulate a flow, how do you impose boundary conditions? 

Now, suppose you have the data, can you recover the Hamiltonian from that? 

Here, we understand how the NN helps achieve that. 

Slide 8, the given example could be the Schr"odinger equation. To assess whether that is the solution, 
you will need to know how to derive your neural network. You assess for the residual, which must 
be zero, in the ideal case. The data fed into the PINN could be the boundary condition or initial condition. 

YOU COULD ACTUALLY TURN THE TWO ELECTRON BEAM INTO A PINN. 

Puedes usar tensorflow, pytorch o jax para lograr eso. Para lograr eso, necesitas entender si es forward o reverse problem. 

Chance si lo hago lo puedo publicar en arxiv! :D 

**NO MAMES CLARO, LE PUEDES PREGUNTAR AL KARNALLLLLL Y LE PUEDES DECIR QUE SI LO PUEDES PUBLICAR CON ELLLLLLLL!!**
* [A Deep-Learning-Based Particle-in-Cell Method for Plasma Simulations](https://www.pdc.kth.se/about/publications/pdc-newsletter-no-2-2021/a-deep-learning-based-particle-in-cell-method-for-plasma-simulations-1.1130171)


"i am using pinn as a formal solver, I am not casting anything in time". 

* Compute the [Eigen-spectral](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix) of "this".
    * Stiff PD


* Numerical fidelity is different from experimental fidelity; how do you account this into real-time input data? 
    * In FEM: you have a primar error; one must keep track of the behaviour of this error in order to adapt the new mesh. Exactly this idea can be implemented in PINNs. 
        * How do you change the basis point w.r.t. residual values? Definition of probability: a bounded function of the residual. Similar idea in PINN. 

* You have residual loss and data loos: how do they impact the convergence.
* PINN is nothing but "constrained optimization": if you come from optimization background, this helps. 


* Lagrange multiplier approach: a constrain for a constrained problem. 
    * PD is the regularizer of the optimization term (in the PINN).
        * The PD term can become more cumbersone the loss function, so it's important to find a balance so that the bias in the algorithm does not increases too much. 

* Understand the error decomposition from the typical machine learning algiruthm 
    * Optimization error and generalization error (generalization error = estimation error + approximation error)
        * One must overcome these eroor to approximate to the trurh. 
    * NN class conly accounts for optimization error + estimation error. 
    * Optimization error does not mean that we reduce it by finding the global minima, but the best local minima. 
    * Optimization error is the most difficult to record: people tend to report only the estimation + approximation errors. There's a group dedicated to understanding the optimization error.

* "Yes, pinns work well for forward problems, but what about inverse problems?"

* Good practice: log the operations that the algorithm perform w.r.t. the important variables. 
    * What is `tensorflow.GradientTape`
    * six-line code to compute the derivative 
    * REMARK: DO NOT USE NUMPY exp(), USE TENSORFLOW exp(); use the "differentiable library". And, of course, the other operators / functions.  

* He showed that only a couple of lines from a regular NN turned it into a PINNS. 

For PINNs, the the vector multiplication to jacobina, the jacobian will have the trace as non-zero, but the other non diagonal values are zero. 

## INVERSE PROBLEMAS 

There's a notebook on a POISSON QEUATION INVERSE SCALAR using PyTorch. 
* `poisson_inverse_function.ipynb`

* How do you sample uniformily? Take the sqrt of the distribution that you are sampling. --> step for colocation point. 
    * Task: change the 0.01 into 1, see how much it changes, 

* Three changes that you have to do, depending on the space/time optimization of your problem. 

* [REFERENCE](https://arxiv.org/abs/2009.04544): Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism
    * converting the function into a min=-max problem . 
    * Minimazion loss with selft-adaptling weights. 
    * First, it looked good; but, there is an issue:
        1. Something failed: the lambda will increase withouth any contraint. 
            * NN is not good for high frequency
            * Solution: force a constrain. 
                * Residual-based attention: like in Finite Element Method. 
            * "If there's variance in the lambda, it means that there's convergence" algo asi, parece que puedes mirar el residuo y saber si esta convergiendo con todo y ruido. 
                * Actually, this is what turned the residual based attention weights; example, the helmholtz equation. 
* INFORMATION BOTTLENECK THEORY
    * Let's look at the signal-to-noise ratio. 
    * Gives the typical learning curve of the PINN. 
        * There are three main stages: fitting, transition, diffusion. Slide 45.
        * This is understood by first having a clear understanding on cross-entropy. 
        * [Georgios C. Anagnostopoulos](https://scholar.google.com/citations?user=P5VBBVAAAAAJ&hl=en)

* Hard constrains OFNeuralN. 

* If the solution has hight frequencies, it is possible that you will make use of Fourier series. ~30 ish, there's a comment on how you turn signal (frequencies) into a feature; this is useful for Helmholz equation, wave equation, etc. 

* Garlekin projection for non-linear PDE 

I wonder if there has been exploration on tensors and PINNs; relativity? 






---

# TOPICO PARA JAGUAR EN LA SELVA

* EMPODERAMIENTO VS AUTONOMIA
    * ¿Quién puede empoderar a quien?
        * Emilia Perez, intento de empoderamiento por parte de europeos???
        * Blancos no pueden empoderar a, por ejemplo, LATAM. 
        * LATAM puede empoderar a LATAM. 
        * Blancos pueden otorgar autonomía a LATAM, pero LATAM necesita empoderamiento para tomar dicha autonomía. 

* OBJETIVO: convencer a vulnerables a tomar y divulgar empoderamiento y convencer a los que ya están empoderados a tomar autonomía. 
    * GENERAL CONCEPT: no es sobre razas, es sobre quién tiene el poder y quién necesita (o se beneficia) del empoderamiento. 


