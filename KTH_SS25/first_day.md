# 1$^{st}$ Day Lecture


### Relevant libraries

* (PyTorch](https://pytorch.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [Jax](https://docs.jax.dev/en/latest/)
* [jit](https://numba.pydata.org/numba-doc/dev/user/jit.html): just-in-time compilation.
    * Two ways of using PyTorch + JIT: `jit.trace` and `jit.script`
* [wandb](https://pypi.org/project/wandb/)

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
    * Example of `__init__` in slide 30. 
* Mention the difference between CPU, GPU, and TPU. 
* Does NNs require CV?
* Can one process real-time input data with these tools? 
* What is a `context manager` in Python? When is it needed? A brief explanation [here](https://book.pythontips.com/en/latest/context_managers.html).
* What is a hook function in programming? 
    * What to do when you want to log-in your model while training?
    * `wandb`: the `git` for your models, kind of. Apparently, it creates logs.
        * Particularly useful for LLMs, for instance. 
        * [doc](https://docs.wandb.ai/)  

## INTRO TO TENSORFLOW

### So, it seems that, `keras` is the Windows and `pytorch` is the linux of deep learning / NNs?

This used to be the "to-go" library as it was the OG but nowadays, there are many more options that do not make `tensorflow` the ideal library to work with; either in academia or industry. Some drawbacks is that it's difficult to debug, and placeholders were requires.Version 2 focused on resolve those issues; it ended up being a bit less efficient but much easier to debug and mantain. 

It introduced `keras` an Application Programming Interface (API) for building and training models. 

* Tensorflow calculates derivatives similar to Pytorch and Jax, via AutoDiff.

* [Gradient Tape](https://www.tensorflow.org/api_docs/python/tf/GradientTape): `tf.GradientTape(
    persistent=False, watch_accessed_variables=True)`

* `Keras` was developed because `tensorflow`'s API was not great. 
    * Thanks to keras, one can get a model overview / summary / etc. 
    * Keras allows you to have more livberty to define parameters in an "easier" way. For instance, setting the loss function, can be done by stating it with a `str`; however, with Torch, one may need to import it. 
    * Easier to implement tensorflow in an arduino instead of PyTorch
    * It appears that PyTorch requires you to be more concious about the space that the model may need etc. It requires more knowledge domain to be implemented effectively. 
    * It can take a couple of seconds to process a 60,000-dataset size.
    * `.fit` =/= `.compile`

* Eager Mode vs Graph Mode
    * [Medium Article](https://jonathan-hui.medium.com/tensorflow-eager-execution-v-s-graph-tf-function-6edaa870b1f1)
    * These are different options to obtain the same result; one is more user-friendly than the other but there is a trade-off with code efficienty and time run. Slide 27 of the last lecture gives an _vanilla_ example of a Eager Mode using the MNIST dataset. 
        * Using the same example with the `keras` implementation, which took 2s to process the data; here, it takes about 10s. The other more takes 1.43s; so, faster than `keras`.

    ### SO THERE ARE THREE DIFFERENT WAYS OF RUNNING AN ALGORITHM with TensorFlow

    * Keras (2s, MNIST Example)
    * Eager Mode (~10s, MNIST Example)
    * Graph Mode (~1.4s, MNIST Example; usually, about 5x faster than Eager mode)
    * What about PyTorch?




