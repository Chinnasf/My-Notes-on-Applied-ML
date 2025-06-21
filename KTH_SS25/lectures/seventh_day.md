# MODULUS Library, K. Shukla

### A quick Intro in GPs

Think: how do you sample from a normal distribution? 
* Metropolis hastling algorithm

What is covariance? A correlation coefficient. Why are covarience in GP symmetric positive? 
"Correlation coefficient is nothing but a distance". 

Why Gaussian Processes fell behind of neural networks? Because of the size of the covariance matrix. 

### Can there be non-positive (symmetric) correlation matrices? 
* Negative definite: no it cannot, because it measures distance. 

Why? because of the definition of the positive-definie definition and the construction of the correlation matrix. 

A bit of the notes he said:
* So one samples **x**~_N(μ, σ)_
* Where **x**$=[x_1,X_2]$ with both being non correlated.

But aaaah we're computing a correlation kernel... 

oxi, I need to learn better this concept. 

BLUE: best linear unbiased estimator

**THERE IS A CODE THAT TEACHS HOW TO SAMPLE FROM A GAUSSIAN PROCESS!**

No correlation = high variance. 

Take a NN, just one layer, with infinity neurons ---> that is a gaussian process. 

* If there were specialized processors units that would focus on computing the inverse of a matrix, there would still be an issue with the inverse of the kernel. "It's not that easy". 
    * I was thinking that, if there were like 'GPUs' but for inverse operations, then GPs would overcome NNs. No? 
* SURROGATES FOR THE INVERSE OF THE MATRICES: that's what he understood of my question.
    * This is why he said "it's not that easy"; I think he thought I wanted a Onet for the inverse of a matrix. ohhhhh. 
    * The inverse of a derivative, I think 

* BLAS (Basic Linear Algebra Subprograms):: seems to be a programming language. This is in every computer. If you go to the source code of torch, for instance, it's written in BLAS. BLAS has thousands of lines to compute a the inverse of a matrix; this is mainly because of the architecture of CPUs. 


## HOW MUCH ARE HUMANS LIMITATED BY THEIR SENSES / TECHONOLOGY?

So, we cannot see wavelengths that are outise $\lambda = [400,600]$nm, so, that's a limitation. Our brain receives thousands of information inputs, but still, can only process a fraction of it. 

Nowadays, we have techniligy, we have AI, like cgatGPT wich depends on trillions of parameters; still, that's limited. 

GPs are a extremelly powerful tool, but we're limitated by their inverse of a matrix. If we could access all processors units of the world to process one GP, which one would it be? To acess hidden information about nature. Just like when the telescopes around the world united to get a photo of a black hole. 

What I like about this proffessor is that he is so passionate about the topics he teach; he also cares about studings following his words and understanding. For him, there are no stupid questions. I wish I could take a full lecture by him. 

Me recuerda a cuando Nati decia que estaba enamorada de turbo jaja. 

Le debería contar mi idea loca... 


# SINGULARITY AND CONTAINERS

Big problems require big resources. 

When you move the data from CPU to GPU, you have to make sure there's no traffic because the latence time is very high. 

Z.B.; if you are working with batches, you can get the maximal performance by being aware how you can manage the sending of the data and what does the code does in the meantime.

Can you treat regular laptors as HPCs? 

[HPC Centers](https://centers.hpc.mil/users/docs/general/singularity.html) say: Singularity is a tool for running software containers on HPC systems, similar to Docker. Singularity is the first containerization technology supported across DSRC HPC resources. In 2021, development of the container technology known as "Singularity" forked resulting in two similar products.

**Make sure you choose your datatype very carefully.**

* SHOWCASE: how to install any package through singularity. 

# PARALLEL DATA

Back propagation es extremelly expensive; hence, models have been developed to avoid this cost. It's called:
* Extending the Forward Forward Algorithm [arXiv:2307.04205](https://arxiv.org/abs/2307.04205)

**GPUs you use, according to your kernel.**

GPUs use their clock to generate their random number, so if you work with parallelism and your model requires a random seed; then, the input seed would be different for all initializations for the slight time variation. All initialization parameters must be the same, this is why broadcasting is needing when working with data parallelism. 


However, it must be that the sensitivity of the parameters should not impact vastly.  