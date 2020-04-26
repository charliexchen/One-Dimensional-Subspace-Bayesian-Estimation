import numpy as np
import abc

from constraints import Constraint

class Objective(object):
    f = None
    constraints = []
    
    def __init__(self,
                 f, 
                 constraints,
                 n_dimensions_in,
                 n_dimensions_out = 1,
                 **kwargs):
        self.f = f
        self.n_dim_in = n_dimensions_in
        self.n_dim_out = n_dimensions_out
        for constraint in constraints: 
            constraint.check_bounds(n_dimensions_in)
        self.constraints = constraints
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
        X = self.__cast(X, self.n_dim_in)
        constraint_eval = np.broadcast_to(self.evaluate_constraints(X), X.shape)
        f_eval = np.ma.apply_along_axis(self.f, 
                                    -1, 
                                    np.ma.array(X, mask = constraint_eval),
                                    **self.params)
        f_eval = np.ma.filled(f_eval, fill_value = np.float64('NaN'))
        if self.n_dim_out == 1: 
            if len(f_eval.shape) + 1 != len(X.shape):
                raise ValueError('Objective function expected single output, got vector of length {}'.format(f_eval.shape[-1]))
            return f_eval[:, np.newaxis]
        else:
            if f_eval.shape[-1] != self.n_dim_out:
                raise ValueError('Objective function expected vector of length {} output, got vector of length {}'.format(self.n_dim_out, 
                                                                                                                    f_eval.shape[-1]))
            return f_eval

    def evaluate_constraints(self, X):
        # Fix edge-case where no constraints exist
        if len(self.constraints) == 0:  return [False]
        eval_constraints = [constraint.evaluate(X) for constraint in self.constraints]
        # Swap constraints on top axis to constraints on bottom axis
        eval_constraints = np.swapaxes(np.array(eval_constraints).astype(int), 0, -1)
        eval_constraints = eval_constraints.sum(axis = -1) > 0
        return eval_constraints[:, np.newaxis]


class Rosenbrock(Objective):
    def __init__(self, 
                 n_dimensions = 2):

        def f(X, **kwargs):
            return np.sum((100*(X[1:]-X[:-1]**2)**2 + (1-X[:-1])**2))

        super().__init__(f, [], n_dimensions_in = n_dimensions, 
                                n_dimensions_out = 1, 
                                **{})

class Eggholder(Objective):
    def __init__(self): 

        def f(X, **kwargs): 
            return -(X[1] + 47)*np.sin(np.sqrt(np.abs(X[0]/2 + (X[1]+47)))) + \
            -X[0]*np.sin(np.sqrt(np.abs(X[0] - (X[1]+ 47))))
             

        def bound_constraint(X, **kwargs):
            return np.all(np.logical_or(X < -512, X > 512))
        
        constraints = [
            Constraint(bound_constraint, **{})
        ]

        super().__init__(f, constraints, 
                            n_dimensions_in = 2, 
                            n_dimensions_out = 1, 
                            **{})

if __name__ == "__main__":
    obj = Rosenbrock(n_dimensions = 2)
    eval_val = obj.evaluate([[[2, 1.5 ], [2, 1.5 ]],
                        [[2, 1.5 ], [2, 1.5 ]]])
    assert np.all(eval_val == 626)
    assert np.array_equal(obj.evaluate([[2, 1.5 ], [2, 1.5 ]]).shape, [2, 1])
    obj = Eggholder()
    assert np.allclose(obj.evaluate([512, 404.2319]), -959.6407, atol = 1e-3)
