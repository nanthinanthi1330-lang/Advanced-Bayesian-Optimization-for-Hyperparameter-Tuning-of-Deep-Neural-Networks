# Short Report

Objective: minimize validation loss on CIFAR-10 using Bayesian Optimization (EI) vs Random Search.

Search space: lr [1e-5,1e-2], batch {32,64,128,256}, dropout [0,0.6], base_filters [16,64], epochs [3,12].

Protocol: fixed budget (30-50 trials), log results to results/, produce convergence plot and final JSON configs.
