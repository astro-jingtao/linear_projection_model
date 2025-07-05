import numpy as np

EPS = 1e-10


def get_M(M, B, X, W, eps=EPS):
    # X = MB
    # W = 1/X_err**2
    return M * np.dot((W * X), B.T) / ((W * M.dot(B)).dot(B.T) + eps)


def get_B(M, B, X, W, eps=EPS):
    # X = MB
    # W = 1/X_err**2
    return B * np.dot(M.T, (W * X)) / ((M.T).dot(W * M.dot(B)) + eps)


def get_B_L(M, B, X, W, grad_pos, grad_neg, alpha=1, eps=EPS):
    # X = MB
    # W = 1/X_err**2
    # dL/dB = grad_pos(B) - grad_neg(B)
    up = np.dot(M.T, (W * X)) + alpha * grad_neg(B)
    low = (M.T).dot(W * M.dot(B)) + alpha * grad_pos(B) + eps
    return B * up / low
