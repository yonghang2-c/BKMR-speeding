# Kernel Comparison: Full BKMR, RFF-only, MR-QMC-IFF

## Introduction

This script compares three kernel-based methods for modeling a continuous outcome y as a function of exposures Z∈ℝⁿˣᵖ and covariates X∈ℝⁿˣᵠ, focusing on capturing non-linear and interaction effects while balancing cost and approximation error.

### 1. Full BKMR

Exact Gaussian-process regression with covariance:

<img src="eq_images/kernel.png" width="300"/>

Fitting the model:

<img src="eq_images/full_model.png" width="300"/>

MCMC via `kmbayes` yields full posterior draws of f, enabling direct inference on exposure interactions.

### 2. RFF-only (Random Fourier Features)

By Bochner’s theorem, any shift-invariant kernel can be expressed as:

<img src="eq_images/bochner.png" width="300"/>

with the spectral density:

<img src="eq_images/p_omega.png" width="300"/>

Defining the random feature map $\phi_{\omega,b}(z)=\sqrt{2}\cos(\omega^\top z + b)$, we have the expectation:

<img src="eq_images/expectation.png" width="300"/>

Approximate via $D$ Monte Carlo samples:

<img src="eq_images/phi_rff.png" width="400"/>

so that:

<img src="eq_images/phi_rff_expect.png" width="400"/>

Monte Carlo approximation of the Gaussian kernel via Bochner’s theorem:

<img src="eq_images/rff_map.png" width="300"/>

Bayesian linear model in feature space:

<img src="eq_images/rff_lin.png" width="300"/>

Cost: 

<img src="eq_images/complexity_rff.png" width="200"/>

Error: 

<img src="eq_images/error_mc.png" width="200"/>

### 3. MR-QMC-IFF

Global Quasi-Monte Carlo features:

<img src="eq_images/mr_global.png" width="300"/>

Combined global and local features:

<img src="eq_images/mr_combined.png" width="300"/>

Error: 

<img src="eq_images/error_qmc.png" width="200"/>, 

variance reduced via clustering, BKMR-level accuracy with fewer features.

## Summary

- **Full BKMR**: exact GP inference:
-
- Cost:
-
- <img src="eq_images/complexity_bkmr.png" width="200"/>.
-
- **RFF-only**: MC kernel approximation:
-
- Cost:
-
- <img src="eq_images/complexity_rff.png" width="200"/>,
-
- Error:
-
- <img src="eq_images/error_mc.png" width="200"/>.
-
- **MR-QMC-IFF**:
- QMC+local features:
-
- Error: <img src="eq_images/error_qmc.png" width="200"/>

scalable to large n and p.
