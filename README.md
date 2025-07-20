# Kernel Comparison: Full BKMR, RFF-only, MR-QMC-IFF

## Introduction

This script compares three kernel-based methods for modeling a continuous outcome y as a function of exposures Z∈ℝⁿˣᵖ and covariates X∈ℝⁿˣᵠ, focusing on capturing non-linear and interaction effects while balancing cost and approximation error.

### 1. Full BKMR

Exact Gaussian-process regression with covariance:

<img src="kernel.png" alt="Gaussian Kernel" width="300" />

Fitting the model:

![Full BKMR Model](full_model.png)

MCMC via `kmbayes` yields full posterior draws of f for interaction inference.

### 2. RFF-only (Random Fourier Features)

Monte Carlo kernel approximation:

![RFF Map](rff_map.png)

Bayesian linear model in feature space:

![RFF Linear Model](rff_lin.png)

Cost: O(nD²), error: O_p(D⁻¹ᐟ²).

### 3. MR-QMC-IFF

Global Quasi-Monte Carlo features:

![MR Global](mr_global.png)

Combined global and local features:

![MR Combined](mr_combined.png)

Error: O(D⁻¹), variance reduced via clustering, BKMR-level accuracy with fewer features.

## Summary

- **Full BKMR**: exact GP inference, O(n³) cost.
- **RFF-only**: MC kernel approximation, O(nD²) cost, O_p(D⁻¹ᐟ²) error.
- **MR-QMC-IFF**: QMC+local features, O(D⁻¹) error, scalable to large n and p.
