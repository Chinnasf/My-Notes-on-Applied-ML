## Optimization Methods for Different Loss Functions

Quick Guide:


 | Path                        | Suggested Method                                         | Notes                                          |
| --------------------------- | -------------------------------------------------------- | ---------------------------------------------- |
| Differentiable + Large Data | **SGD, Adam, RMSProp**                                   | Standard for deep learning                     |
| Differentiable + Small Data | **L-BFGS, BFGS, Newton**                                 | Good for convex problems                       |
| Non-Diff + Expensive        | **Bayesian Optimization**                                | Great for tuning black-box models              |
| Non-Diff + Cheap            | **Genetic Algorithms, Simulated Annealing, Nelder-Mead** | Useful when gradients don’t exist or are noisy |


Decision Tree:

```
                   ┌────────────────────────────────────────┐
                   │   Do you have a differentiable loss?   │
                   └────────────────────────────────────────┘
                                 │
                        ┌────────┴────────┐
                        │                 │
                      YES                NO
                        │                 │
        ┌─────────────────────────┐   ┌────────────────────────────┐
        │ Do you have many data   │   │ Is the function expensive  │
        │ points (e.g. ML models)?│   │ to evaluate? (e.g. tuning) │
        └─────────────────────────┘   └────────────────────────────┘
                  │                              │
           ┌──────┴──────┐                  ┌────┴────┐
           │             │                  │         │
         YES            NO                YES        NO
           │             │                  │         │
 ┌────────────────┐  ┌────────────────────┐┌────────┐ ┌─────────────────┐
 │ Use SGD,       │  │ Use BFGS / L-BFGS  ││ Use    │ │ Use evolutionary│
 │ Adam, RMSProp  │  │ or Newton method   ││Bayesian│ │ or heuristic    │
 │ (first-order)  │  │ (second-order)     ││opt.    │ │ methods (GA, SA)│
 └────────────────┘  └──────────┬─────────┘└────────┘ └─────────────────┘
                                │
                                │
                                ▼
                    ┌────────────────────────────┐
                    │ Do you expect sparsity in  │
                    │ the optimal parameters?    │
                    └────────────────────────────┘
                                │
                         ┌──────┴──────┐
                         │             │
                       YES            NO
                         │             │
              ┌──────────────────┐   ┌──────────────────┐
              │ Use Coordinate   │   │ Stick with BFGS  │
              │ Descent (e.g.    │   │ or first-order   │
              │ LASSO, L1-penal) │   │ methods          │
              └──────────────────┘   └──────────────────┘

```

