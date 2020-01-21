import matplotlib.pyplot as plt
import numpy as np
import torch
import kernels
import constraints

class GaussianProcess(object):
    
    kernel = None
    sigma = None
    inverse = False
    labels = False

    def __init__(self, kernel, sigma):
        if not isinstance(kernel, kernels.Kernel):
            raise TypeError('Kernel is not a valid kernels.Kernel class')
        self.kernel = kernel
        self.sigma = sigma
    
    def fit(self, X, y):
        y = self.kernel.cast(y)
        X = self.kernel.cast(X)
        if y.shape[-1] != 1:
            raise ValueError('Expected vector of dimension 1, but got vector of dimension {}'.format(y.shape[-1]))

        self.kernel.fit(X)
        self.inverse = torch.cholesky_inverse(self.kernel.kernel + torch.eye(self.kernel.n_dim_in)*(self.sigma**2))
        self.labels = y
        return self

    def predict(self, X):
        _, kernel_increment, kernel_predict , _ = self.kernel.predict_increment(X)
        k_star_k, k_star_k_star = kernel_increment
        mean = torch.mm(torch.mm(torch.transpose(k_star_k, 0, 1), self.inverse), self.labels)
        covariance = k_star_k_star + torch.eye(self.kernel.n_dim_in)*(self.sigma**2) - \
                     torch.mm(torch.mm(torch.transpose(k_star_k, 0, 1), self.inverse), k_star_k)
        return mean, covariance

if __name__ == "__main__":
    x = np.linspace(0, 2*np.pi, 50).reshape((-1, 1))
    x_test = (np.random.random((20, 1))*2*np.pi).reshape((-1, 1))
    y = np.sin(x)
    gp = GaussianProcess(
        kernel = kernels.SquaredExponential(0.1), 
        sigma = 0.05
    ).fit(x, y)
    y_hat_mean, y_hat_covariance = gp.predict(x_test)
    plt.plot(x, y, 'r.')
    plt.plot(x_test, y_hat_mean, 'b.')
    plt.show()
    pass