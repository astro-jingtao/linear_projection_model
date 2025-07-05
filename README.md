# Linear Projection Model

Collection of linear projection model, which try to express the data $X$ as the linear projection of latent variable $Z$, as

$$
X = AZ
$$

## Implemented Methods

- `ica.py`
  - Rotation based non-negative ICA. Force $Z$ to be non-negative, not both $A$ and $Z$ as NMF.
  - Reference
    - [Algorithms for nonnegative independent component analysis](http://ieeexplore.ieee.org/document/1199651/)
- `lda.py`
  - LDA loop to circulate `n_components < n_class - 1` constrain
- `nmf.py`
  - Weighted NMF, Euclidean distance based update rule; It considers cell-wise weight, i.e. weight $w_{ij}$ for each element of $X$.
  - It can deal with additional loss $L(Z)$
  - Reference
    - Weighted version: [Weighted nonnegative matrix factorization](https://ieeexplore.ieee.org/abstract/document/4959890) 
    - Additional loss: [Adaptive weights for NMF with additional priors](http://ieeexplore.ieee.org/document/7432744/)
- `jax/nmf.py`
  - Same as `nmf.py` but for `JAX` version
  - Additional support:
    - Get the gradient of $L(Z)$ by automatic differentiation

## Notes

- If you are looking for weighted PCA, please refer to https://github.com/jakevdp/wpca

## Acknowledgment

- Non-negative ICA is based on [Marius1311's implementation](https://github.com/Marius1311/Non-negative-ICA).