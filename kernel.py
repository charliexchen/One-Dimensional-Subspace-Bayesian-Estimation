import torch
import numpy as np

class Kernel(object): 
    f_pairwise = None
    f_vectorized = None
    X = None
    kernel = None
    params = None
    n_dim_in = None
    def __init__(self, f_pairwise, 
                       f_vectorized = None,
                       n_dimensions_in = None,
                       **kwargs):
        self.f_pairwise = f_pairwise
        self.f_vectorized = f_vectorized
        self.params = kwargs
        self.n_dim_in = n_dimensions_in
        self.__validate_kernel()
    
    def __validate_kernel(self):
        test_feats = None
        if self.n_dim_in is not None: 
            test_feats = torch.empty(5, self.n_dim_in).uniform_(0, 1)
        else:
            test_feats = torch.empty(5, 3).uniform_(0, 1)
        matrix = self.__evaluate_matrix(self.cast(test_feats))
        if not (torch.abs(matrix - matrix.transpose(0, 1)) < 1e-8).all():
            raise ValueError('Kernel matrix is not symmetric. Review kernel function')
        # If matrix is Hermitian (as K(i, j) = K(j, i)), all eigenvalues should be real
        if (torch.eig(matrix)[0][:, 0] < 0).any(): # Check if matrix is positive semi-definite
            raise ValueError('Kernel matrix has negative eigenvalue and is not positive semi-definite. Review kernel function')

    def __evaluate_matrix(self, X):
        if self.f_vectorized is None:
            matrix = torch.empty(X.shape[0], X.shape[0])
            for i in range(X.shape[0]):
                for j in range(i+1):
                    i_j_cov = self.f_pairwise(X[i, :], X[j, :], self.params) # Pairwise
                    matrix[i, j] = i_j_cov
                    matrix[j, i] = i_j_cov
            return matrix
        else:
            return self.f_vectorized(X, self.params)

    def __increment_matrix(self, X):
        matrix = torch.empty(X.shape[0], X.shape[0])
        matrix[:-1, :-1] = self.kernel
        for i in range(X.shape[0]):
            i_j_cov = self.f_pairwise(X[i, :], X[-1, :], self.params) 
            matrix[-1, i] = i_j_cov
            matrix[i, -1] = i_j_cov
        return matrix

    def cast(self, X): 
        # Cast from numpy array
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        # Cast from native python tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        if X.dtype != torch.float32:
            X = X.type(torch.FloatTensor)
        return X

    def fit(self, X):
        X = self.cast(X)
        if self.n_dim_in is None: 
            self.n_dim_in = X.shape[-1]

        if len(X.shape) != 2: 
            raise ValueError('Expected matrix input but received tensor of order {} and shape {}'.format(len(X.shape), X.shape))
        if (self.n_dim_in is not None) and (X.shape[-1] != self.n_dim_in):
            raise ValueError('Expected features of dimension {}, got features of dimension {}'.format(self.n_dim_in, X.shape[-1]))
         
        self.kernel = self.__evaluate_matrix(X)
        self.X = X
        print(self.kernel)
     
    def fit_increment(self, X_inc):
        if self.kernel is None: 
            raise ValueError("Cannot run fit_increment before running fit on training data")
        X_inc = self.cast(X_inc)
        
        if len(X_inc.shape) != 1: 
            raise ValueError('Expected vector input but received tensor of order {} and shape {}'.format(len(X_inc.shape), X_inc.shape))
        if (X_inc.shape[-1] != self.n_dim_in):
            raise ValueError('Expected vector of dimension {}, got vector of dimension {}'.format(self.n_dim_in, X_inc.shape[-1]))

        X_new = torch.cat([self.X, X_inc.unsqueeze(0)], 0)
        self.kernel = self.__increment_matrix(X_new)
        self.X = X_new
        print(self.kernel)

class SquaredExponentialPairwise(Kernel):

    def __init__(self, 
                 tau):
        tau = torch.squeeze(super().cast(tau))

        if not len(tau.shape) == 0:
            raise ValueError('Invalid shape for tau - expected scalar, got tensor of shape {}'.format(tau.shape))

        def f_pairwise(x_1, x_2, params):
                return torch.exp(-1/(2*params['tau']**2) * torch.norm(x_1 - x_2, p = 'fro'))

        super().__init__(f_pairwise = f_pairwise, 
                         f_vectorized= None,
                         n_dimensions_in = None,
                         **{'tau': tau})

class SquaredExponentialVectorized(Kernel):

    def __init__(self, 
                 tau):
        tau = torch.squeeze(super().cast(tau))

        if not len(tau.shape) == 0:
            raise ValueError('Invalid shape for tau - expected scalar, got tensor of shape {}'.format(tau.shape))

        def f_pairwise(x_1, x_2, params):
                return torch.exp(-1/(2*params['tau']**2) * torch.norm(x_1 - x_2, p = 'fro'))
            
        def f_vectorized(X, params):
                X = X.permute(1, 0)
                raise NotImplementedError('Lol not enough time')
                X = X.permute(0, 1)
                pass

        super().__init__(f_pairwise = f_pairwise, 
                         f_vectorized= None,
                         n_dimensions_in = None,
                         **{'tau': tau})

class ARDSquaredExponentialPairwise(Kernel):

    def __init__(self,
                 n_dimensions_in,
                 sigma = 1,
                 lengthscale = 1):
                 
        lengthscale = torch.squeeze(super().cast(lengthscale))
        sigma = torch.squeeze(super().cast(sigma))

        if not len(sigma.shape) == 0:
            raise ValueError('Invalid shape for sigma - expected scalar, got tensor of shape {}'.format(sigma.shape))
        
        if not ((len(lengthscale.shape) == 1) and (lengthscale.shape[0] == n_dimensions_in)):
            raise ValueError('Invalid shape for lengthscale - expected tensor of shape [{}] (n_dimensions_in), got tensor of shape {}'.format(n_dimensions_in, lengthscale.shape))

        def f_pairwise(x_1, x_2, params):
            norm = torch.dot(torch.pow(x_1 - x_2, 2), 1/params['lengthscale'])
            return params['sigma']**2 * torch.exp(-1/2 * norm)

        super().__init__(f_pairwise = f_pairwise, 
                         f_vectorized= None,
                         n_dimensions_in = n_dimensions_in,
                         **{'sigma': sigma, 'lengthscale':lengthscale})


if __name__ == '__main__':
    SE_pw_test = SquaredExponentialPairwise(tau = 4)
    test_feat = torch.empty(5, 4).uniform_(0, 1)
    add_feat = torch.empty(4).uniform_(0, 1)
    SE_pw_test.fit(test_feat)
    SE_pw_test.fit_increment(add_feat)

    ARD_SE_pw_test = ARDSquaredExponentialPairwise(sigma = 5, lengthscale= [1, 2, 3, 4, 5], n_dimensions_in = 5)
    test_feat = torch.empty(3, 5).uniform_(0, 1)
    add_feat = torch.empty(5).uniform_(0, 1)
    ARD_SE_pw_test.fit(test_feat)
    ARD_SE_pw_test.fit_increment(add_feat)
