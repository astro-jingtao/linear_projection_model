import numpy as np
from scipy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def lda_loop(X, y, n_loop=2):
    basics_proj = np.zeros((X.shape[1], n_loop))
    X_proj = X.copy()
    basics_null = np.eye(X.shape[1])
    for i in range(n_loop):
        lda = LinearDiscriminantAnalysis(n_components=1, solver='eigen')
        lda.fit(X_proj, y)
        lda_vec = lda.scalings_[:, 0:1]  # type: ignore
        lda_vec = lda_vec / np.linalg.norm(lda_vec)  # type: ignore
        basics_proj[:, i] = (basics_null @ lda_vec)[:, 0]
        # print(lda_null.shape, linalg.null_space(lda.scalings_[:, 0:1].T).shape)
        basics_null = basics_null @ linalg.null_space(lda_vec.T)
        X_proj = X @ basics_null
    return basics_proj, basics_null
