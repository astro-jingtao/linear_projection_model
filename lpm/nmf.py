import numpy as np


def get_M(M, B, X, W):
    # U -> M
    # V -> B
    # W = 1/X_err**2
    return M * np.dot((W * X), B.T) / (W * M.dot(B)).dot(B.T)


def get_B(M, B, X, W):
    # U -> M
    # V -> B
    # W = 1/X_err**2
    return B * np.dot(M.T, (W * X)) / ((M.T).dot(W * M.dot(B)))