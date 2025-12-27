# Bayesian Inference from First Principles

This repository contains a **from-scratch implementation of Bayesian logistic regression**, focusing on numerical optimization, posterior approximation, and predictive inference without relying on probabilistic programming frameworks.

The primary goal is **conceptual clarity and correctness**, not production readiness.

**Technical write-up:**  
https://cooperniebuhr.github.io/bayesian-inference-from-scratch/

---

## Overview

The current implementation demonstrates:

- Logistic regression with a Gaussian prior  
- Maximum a posteriori (MAP) estimation via Newton’s method
- Posterior uncertainty via the Laplace approximation
- Monte Carlo posterior predictive inference  
- Validation against PyMC (in the accompanying write-up)

All core computations (log-posterior, gradients, Hessians) are implemented explicitly using NumPy.

---

## Repository structure

```text
.
├── bayes_fp.py        # Core implementation (MAP, Laplace, prediction)
├── docs/              # Rendered Quarto HTML (GitHub Pages)
├── README.md
```

At the moment, `bayes_fp.py` is the main artifact. It contains:

- numerically stable logistic and softplus functions  
- Gaussian prior (log-density, gradient, Hessian)  
- Bernoulli logistic likelihood (log, grad, Hessian)  
- Newton solver for MAP estimation  
- Laplace covariance construction  
- Monte Carlo predictive inference  

---

## Code highlights

### MAP estimation

The MAP estimate is obtained by maximizing the log posterior using Newton’s method with backtracking line search:

```python
w_hat, H_hat, info = newton_map_logistic(X, y, sigma0=1.0)
```

The Hessian at the MAP is reused for the Laplace approximation.

---

### Laplace approximation

Posterior uncertainty is approximated as:

p(w | D) ≈ N(w_MAP, -H⁻¹)

```python
Sigma = laplace_covariance(H_hat)
```

---

### Posterior predictive inference

Predictions marginalize over parameter uncertainty using Monte Carlo integration:

```python
probs = predictive_laplace_mc(X_new, w_hat, Sigma)
```

---

## Scope and non-goals

This repository **is not**:

- a general-purpose Bayesian regression library  
- a replacement for PyMC, Stan, or NumPyro  
- optimized for large-scale or high-dimensional problems  

It is primarily:

- a transparent reference implementation  
- a numerical sanity check for Bayesian GLMs  
- a foundation for extensions (hierarchical models, alternative priors, HMC)
- a learning tool for me

---

## Future directions

Possible extensions:

- Hierarchical priors  
- Non-Gaussian priors  
- Full HMC sampling  
- Multiclass logistic regression  
- Comparison with variational inference  

---

## Notes

This project is written as a **technical implementation note**, not a tutorial.  
Familiarity with Bayesian inference, numerical optimization, and linear algebra is assumed.
