import numpy as np

class Constraint(object): 
    f = None
    params = None
    n_dim = None
    bounds_checked = False
    def __init__(self, 
                 f,
                 **kwargs):
        self.f = f
        self.params = kwargs

    def __cast(self, X, req_shape):
        if isinstance(X, list): 
            X = np.array(X)
        if X.shape[-1] != req_shape:
            raise ValueError('X is of base length {}, \
                             but expected base vector of length {}'.format(
                                 X.shape[-1], 
                                 req_shape))
        # Handle single-vector case
        if len(X.shape) == 1:
            X = X[np.newaxis, :]
        return X

    def evaluate(self,
                 X):
        if self.n_dim is not None: 
            X = self.__cast(X, self.n_dim)
        f_eval = np.apply_along_axis(self.f, -1, X, **self.params)
        return f_eval

    def check_bounds(self, n_dim):
        bound_check = self.__cast(np.zeros(n_dim), n_dim)
        if self.evaluate(bound_check).dtype != 'bool':
            raise ValueError('Constraint invalid - does not produce boolean output')
        self.n_dim = n_dim
