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
* **AUTOMATED DIFFERENTIATION** This is required a lot for this school; provided by `jax`: `jax.grad()`.

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
    * [**PARTIAL ANSWER**](https://www.liquidweb.com/gpu/vs-tpu/): A TPU (tensor processing unit) is a specialized chip designed for deep learning and artificial intelligence. Unlike GPUs, which handle a variety of parallel computing tasks, TPUs focus exclusively on processing tensor operations, the core computations in neural networks.
    * Google developed TPUs to accelerate AI workloads efficiently, particularly for machine learning models running on TensorFlow. These chips power everything from language models to image recognition systems, offering high performance with lower energy consumption than traditional GPUs.
* Does NNs require CV?
* Can one process real-time input data with these tools? 
* What is a `context manager` in Python? When is it needed? A brief explanation [here](https://book.pythontips.com/en/latest/context_managers.html).
* What is a hook function in programming? 
    * What to do when you want to log-in your model while training?
    * `wandb`: the `git` for your models, kind of. Apparently, it creates logs.
        * Particularly useful for LLMs, for instance. 
        * [doc](https://docs.wandb.ai/)  
* **WHAT IS A PURE FUNCTION?**
    1. Always gives the same output for the same input 
        * It doesn't rely on or modify anything outside its scope.
    2. Has no side effects. It does not:
        * change global variables
        * modify input arguments
        * perform I/O (e.g. print, write to disk)
        * use random values unless passed explicitly
    3. EXAMPLE of a pure function:
        * Always gives the same result.
        * Doesn't change anything outside.
        * Doesn't depend on external state.
        * ```Python
            def add(a, b):
                return a + b
        ```
    4. EXAMPLE of an impure function: 
        * Depends on the external variable `x`.
        * If `x` changes, the output changes.
        * Modifies the input list `lst` directly --> side effect!
        ```Python
            x = 10

            def add_to_x(b):
                return x + b
            
            def increment_list(lst):
                for i in range(len(lst)):
                    lst[i] += 1
                return lst
        ```

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
    * [Decorators](https://www.tensorflow.org/guide/function) on top of functios; z.B:
         
         ```Python
         @tf.function ## <-- adding this speeds up the code!
         def train_step(
            ...
         )
         return something
         ```

         Not all functions can benefit for the decorations. Apparently, this is something only from tensorflow, not something that would be found in other python libraries. 
    
    * Fun fact: the graph mode **does not allow if statements in the code**, instead, one must use `tf.bool()` or something like that.

### SO THERE ARE THREE DIFFERENT WAYS OF RUNNING AN ALGORITHM with TensorFlow

* Keras (2s, MNIST Example)
* Eager Mode (~10s, MNIST Example)
* Graph Mode (~1.4s, MNIST Example; usually, about 5x faster than Eager mode)
* What about PyTorch?


## JAX Tutorial

Designed for complex scientific computing or custom gradients.

**IMPORTANT**: Mutability: PyTorch allows changing values in place (e.g., weights, buffers), but JAX is built around pure functions and immutability.


```Python

import jax
import jax.numpy as jnp

x = jnp.arange(5)

x.devices() # can be used in multiple devices / GPUs. 
```

* It has been proved beneficial for Bayesian Methods for optimizing the use of random methods. Also for initializing and reproducibility. 
* Have a look at the tutorial. :v 
    * Alright, it seems that people like `jax` not only because of the improved performance, but also because it is traceable.
* `jax` cannot compile conditionals, for instance `if` or `while`. Booleans cannot be processed by jax, apparently because of the way in which data is stored.
    * Have a look at `Marking arguments as static` in the  jpnb.
* It is almost as if you had to learn a new way of thinking for correctly implementing `jax` although it is written in python.



