
# Different PINNs

### Gradient-Enhanced PINN

One can also formulate the inverse problem. 

Advantages with specific application, z.B.: Schrodinger eq. Think about limitations as well. 

### Fractional PINN

Do you remember fractional calculus? 

One application is the fractional diffusion-reaction (Gray-Scott) model. 

There's an 'issue' with these problems, since automatic differentiation cannot be used for this PINN. 
Thus, one must change the collocation points to a random positioning of points. 

### Bayesian-PINN (B-PINN)

One can have a closer look at the uncertainty: discern between epistemic and random. 

Reference: [B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data](https://www.sciencedirect.com/science/article/pii/S0021999120306872)

A comparsion between all the possible PINNs so far; z.B. 

### Conservative-PINN (cPINN)

Makes sure that conservation laws are preserved. 

[Conservative physics-informed neural networks on discrete domains for conservation laws: Applications to forward and inverse problems](https://www.sciencedirect.com/science/article/pii/S0045782520302127)

* "cPINN is only for space-domain decomposition."

* "xPINN is an extension of the cPINN as it can be used for time-domain decomposition as well."

* "$PINN, the combination of BPINN with cPINN", developed in KTH, last year.
    * Could deal with noisy data.
    * It has been used for Fokker–Planck equation.

### PINN for Domain decomposition

[Machine learning and domain decomposition methods -- a survey](https://arxiv.org/abs/2312.14050)

Some issues with PINNs are the uncertainty and the scalability. 


**REMARKS**

* Remember that you can also have a look at the different induced errors/ uncertainty: numerical, random, epistemic, etc. 


**QUESTIONS**

* What are collocation points in these contexts?