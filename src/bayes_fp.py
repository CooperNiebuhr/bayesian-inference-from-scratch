# First principles Bayesian Inference:
# numerics and a MAP via Newton

import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Element-wise sigmoid function."""
    z = np.asarray(z)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out

def softplus(z: np.ndarray) -> np.ndarray:
    """Stable log(1 + exp(z)) element-wise."""
    z = np.asarray(z, dtype=float)

    # log(1+exp(z)) = max(z,0) + log1p(exp(-|z|))
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))

   
# ----- Prior ----- #
 # N(0, sigma0^2 I) #

def log_prior_gaussian(w: np.ndarray, sigma0: float) -> float:
    """Log N(0, sigma0^2 I) density."""
    w = np.asarray(w, dtype=float)
    d = w.size
    return -0.5 * (np.dot(w, w) / (sigma0**2)) - 0.5 * d * np.log(2 * np.pi * (sigma0**2))

def grad_log_prior_gaussian(w: np.ndarray, sigma0: float) -> np.ndarray:
    """Gradient of log prior wrt w."""
    return -(w / (sigma0**2))

def hess_log_prior_gaussian(d: int, sigma0: float) -> np.ndarray:
    """Hessian of log prior wrt w."""
    return -np.eye(d) / (sigma0**2)

# ----- Likelihood ----- #
 # Bernoulli with sigma(Xw) #
def loglikelihood_logistic(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    Log likelihood sum_i [y_i * (x_i^T * w) - softplus(x_i^T * w)]
    """
    z = X @ w
    return float(np.sum(y * z - softplus(z)))

def grad_loglikelihood_logistic(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Gradient wrt w: X^t * (y - sigma(Xw))
    """
    z = X @ w
    p = sigmoid(z)
    return X.T @ (y - p)

def hess_loglikelihood_logistic(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Hessian wrt w: -X^t * W * X, where W is diag(sigma(Xw) * (1 - sigma(Xw)))
    """
    z = X @ w
    p = sigmoid(z)
    w_diag = p * (1.0 - p)
    return -(X.T @ (X * w_diag[:, None]))

# ----- Posterior ----- #

def log_posterior(w: np.ndarray, X: np.ndarray, y: np.ndarray, sigma0: float) -> float:
    """Log posterior up to a constant."""
    return log_prior_gaussian(w, sigma0) + loglikelihood_logistic(w, X, y)

def grad_log_posterior(w: np.ndarray, X: np.ndarray, y: np.ndarray, sigma0: float) -> np.ndarray:
    """Gradient of log posterior wrt w."""
    return grad_log_prior_gaussian(w, sigma0) + grad_loglikelihood_logistic(w, X, y)

def hess_log_posterior(w: np.ndarray, X: np.ndarray, sigma0: float) -> np.ndarray:
    """Hessian of log posterior wrt w."""
    d = X.shape[1]
    return hess_log_prior_gaussian(d, sigma0) + hess_loglikelihood_logistic(w, X)


# ----- MAP via Newton ----- #

def newton_map_logistic(
    X: np.ndarray,
    y: np.ndarray,
    sigma0: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 50,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Find MAP w_hat via Newton's method.
    Returns:
        w_hat: MAP estimate
        H: Hessian at MAP
        info: dict with 'n_iter' and 'converged' keys
    """
    d = X.shape[1]
    w = np.zeros(d, dtype=float)
    info = {"num_iter": 0, "converged": False, "logpost_hist": []}

    def solve_newton_step(H: np.ndarray, g: np.ndarray) -> np.ndarray:
        # Solve H * delta = -g
        # Here H is a Hessian of log posterior. step = -inv(H) * g should go uphill.
        # Fall back to pinv if H is singular.
        try:
            p = np.linalg.solve(H, -g)
            return p
        except np.linalg.LinAlgError:
            # If H is singular, use pseudo-inverse
            return -np.linalg.pinv(H) @ g

    for i in range(1, max_iter + 1):
        lp = log_posterior(w, X, y, sigma0)
        info["logpost_hist"].append(lp)
        g = grad_log_posterior(w, X, y, sigma0)
        H = hess_log_posterior(w, X, sigma0)

        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            info["num_iter"] = i
            info["converged"] = True
            break

        step = solve_newton_step(H, g)

        t = 1.0
        while t > 1e-8:
            w_new = w + t * step
            lp_new = log_posterior(w_new, X, y, sigma0)
            if lp_new >= lp:
                w = w_new
                break
            t *= 0.5
        
        if verbose:
            print(f"iter {i:0d} lp={lp:.6f} ||step||={np.linalg.norm(step):.3e} t={t:.2e}")

        if np.linalg.norm(t * step) < tol:
            info["num_iter"] = i
            info["converged"] = True
            break

    if not info["converged"]:
       info["num_iter"] = max_iter
        
    H_hat = hess_log_posterior(w, X, sigma0)
    return w, H_hat, info

# ---- Sampling methods ----- #
def laplace_covariance(H_at_map: np.ndarray) -> np.ndarray:
    """Covariance from Laplace approximation at MAP."""
    return -np.linalg.inv(H_at_map)

def sample_laplace_posterior(rng: np.random.Generator, w_hat: np.ndarray, Sigma: np.ndarray, nsamples: int) -> np.ndarray:
    """Draw samples from Laplace approximation N(w_hat, Sigma)."""
    return rng.multivariate_normal(mean=w_hat, cov=Sigma, size=nsamples)

def predictive_laplace_mc(Xnew: np.ndarray, w_hat: np.ndarray, Sigma: np.ndarray, nsamples: int = 5000, seed: int | None = None) -> np.ndarray:
    """Monte Carlo predictive probabilities using Laplace approximation."""
    rng = np.random.default_rng(seed)
    w_samples = sample_laplace_posterior(rng, w_hat, Sigma, nsamples)
    logits = Xnew @ w_samples.T  # shape (n_new, nsamples)
    probs = sigmoid(logits)
    return np.mean(probs, axis=1)  # shape (n_new,)