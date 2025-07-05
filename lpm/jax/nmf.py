from jax import jit, grad
from jax import numpy as jnp

EPS = 1e-10

@jit
def get_M(M, B, X, W, eps=EPS):
    # X = MB
    # W = 1/X_err**2
    return M * jnp.dot((W * X), B.T) / ((W * M.dot(B)).dot(B.T) + eps)

@jit
def get_B(M, B, X, W, eps=EPS):
    # X = MB
    # W = 1/X_err**2
    return B * jnp.dot(M.T, (W * X)) / ((M.T).dot(W * M.dot(B)) + eps)


def get_B_L(M, B, X, W, grad_pos, grad_neg, alpha=1, eps=EPS):
    # X = MB
    # W = 1/X_err**2
    # dL/dB = grad_pos(B) - grad_neg(B)
    up = jnp.dot(M.T, (W * X)) + alpha * grad_neg(B)
    low = (M.T).dot(W * M.dot(B)) + alpha * grad_pos(B) + eps
    return B * up / low


def generate_get_B_L(L):
    # X = MB
    # W = 1/X_err**2

    L_grad = grad(L)

    @jit
    def grad_pos(B):
        return jnp.clip(L_grad(B), 0, None)

    @jit
    def grad_neg(B):
        return -jnp.clip(L_grad(B), None, 0)

    @jit
    def _get_B_L(M, B, X, W, alpha=1, eps=EPS):
        # X = MB
        # W = 1/X_err**2
        # dL/dB = grad_pos(B) - grad_neg(B)
        up = jnp.dot(M.T, (W * X)) + alpha * grad_neg(B)
        low = (M.T).dot(W * M.dot(B)) + alpha * grad_pos(B) + eps
        return B * up / low

    return _get_B_L
