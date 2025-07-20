# Kernel Comparison: Full BKMR, RFF-only, MR-QMC-IFF

## Introduction

This script compares three kernel-based methods for modeling a continuous outcome y as a function of exposures Z∈ℝⁿˣᵖ and covariates X∈ℝⁿˣᵠ, focusing on capturing non-linear and interaction effects while balancing cost and approximation error.

### 1. Full BKMR

Exact Gaussian-process regression with covariance:

<img src="eq_images/kernel.png" width="200"/>

Fitting the model:

<img src="eq_images/full_model.png" width="200"/>

MCMC via `kmbayes` yields full posterior draws of f for interaction inference.

### 2. RFF-only (Random Fourier Features)

Monte Carlo kernel approximation:

<img src="eq_images/rff_map.png" width="200"/>

Bayesian linear model in feature space:

<img src="eq_images/rff_lin.png" width="200"/>

Cost: <img src="eq_images/complexity_rff.png" width="100"/>, error: <img src="eq_images/error_mc.png" width="100"/>.

### 3. MR-QMC-IFF

Global Quasi-Monte Carlo features:

<img src="eq_images/mr_global.png" width="200"/>

Combined global and local features:

<img src="eq_images/mr_combined.png" width="200"/>

Error: <img src="eq_images/error_qmc.png" width="100"/>, variance reduced via clustering, BKMR-level accuracy with fewer features.

## Summary

- **Full BKMR**: exact GP inference, <img src="eq_images/complexity_bkmr.png" width="100"/> cost.
- **RFF-only**: MC kernel approximation, <img src="eq_images/complexity_rff.png" width="100"/> cost, <img src="eq_images/error_mc.png" width="100"/> error.
- **MR-QMC-IFF**: QMC+local features, <img src="eq_images/error_qmc.png" width="100"/> error, scalable to large n and p.
