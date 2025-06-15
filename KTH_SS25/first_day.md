# 1$^{st}$ Day Lecture


### Relevant libraries

* (PyTorch](https://pytorch.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [Jax](https://docs.jax.dev/en/latest/)
* [jit](https://numba.pydata.org/numba-doc/dev/user/jit.html): just-in-time compilation.
    * Two ways of using PyTorch + JIT: `jit.trace` and `jit.script`

### Important concepts

* [Computational Graphs](https://www.geeksforgeeks.org/deep-learning/computational-graphs-in-deep-learning/)
    * PyTorch creates these graphs in an improved manner. 
* Data Structures --> important to understand the different types; z.B.: `torch.eye()`
* [Contigious vs non-contigious tensor](https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107)
    * An operation may call another position in the RAM / place copies in other places, etc.
* [Automatic Differentiation](https://www.youtube.com/watch?v=wG_nF1awSSY): 
    * Forward Pass $f(a,b) = a * (a+b)$ 
    * Reverse pass
    * [TU Wien explanation](https://www.youtube.com/watch?v=R_m4kanPy6Q)
    * Useful to compute the gradients
* Function Approximation in NNs
* Graph mode efficiency --> `git`

### GPU Processess

**TORCH**
* Count for time: `%time` within torch (?)
* One can do einstein summation in a fancy way :D
    * `torch.einsum("bij, bjk -> bik", A, B)`
    * [documentation](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html) 

### QUESTIONS

* What does `def __init__()` is?

## INTRO TO TENSORFLOW

This used to be the "to-go" library as it was the OG but nowadays, there are many more options that do not make `tensorflow` the ideal library to work with; either in academia or industry. Some drawbacks is that it's difficult to debug, and placeholders were requires.Version 2 focused on resolve those issues; it ended up being a bit less efficient but much easier to debug and mantain. 

It introduced `keras` an Application Programming Interface (API) for building and training models. 
