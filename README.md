# Linear Projection Model

Collection of linear projection model, which try to express the data $X$ as the linear projection of latent variable $Z$, as

$$
X = AZ
$$

## Implemented Methods

- `ica.py`
  - Rotation based non-negative ICA
    - [Algorithms for nonnegative independent component analysis](http://ieeexplore.ieee.org/document/1199651/)
- `lda.py`
  - LDA loop to circulate `n_components < n_class - 1` constrain
- `nmf.py`
  - Weighted NMF, update rule for Euclidean distance version
    - [Weighted nonnegative matrix factorization](https://ieeexplore.ieee.org/abstract/document/4959890)

## Acknowledgment

- Non-negative ICA is based on [Marius1311's implementation](https://github.com/Marius1311/Non-negative-ICA).