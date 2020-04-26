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
        _, kernel_increment, _ , _ = self.kernel.predict_increment(X)
        k_star_k, k_star_k_star = kernel_increment
        mean = torch.mm(torch.mm(torch.transpose(k_star_k, 0, 1), self.inverse), self.labels)
        covariance = k_star_k_star + torch.eye(self.kernel.n_dim_in)*(self.sigma**2) - \
                     torch.mm(torch.mm(torch.transpose(k_star_k, 0, 1), self.inverse), k_star_k)
        return mean, covariance

if __name__ == "__main__":
    # x = np.linspace(0, 2*np.pi, 50).reshape((-1, 1))
    x = (np.random.random((15, 1))*2*np.pi).reshape((-1, 1))
    x_test = np.linspace(-1, 2*np.pi+1, 50).reshape((-1, 1))
    y = np.sin(x)
    gp = GaussianProcess(
        kernel = kernels.SquaredExponential(tau = 0.5, sigma = 1, use_pairwise_only = False), 
        sigma = 0
    ).fit(x, y)
    y_hat_mean, y_hat_covariance = gp.predict(x_test)
    y_hat_std_diag = torch.diag(y_hat_covariance)
    plt.plot(x, y, 'r.')
    plt.plot(x_test, y_hat_mean, 'b-')
    plt.plot(x_test, np.sin(x_test), 'r-')
    plt.fill_between(x_test.squeeze(), 
                    y_hat_mean.squeeze() + 2*y_hat_std_diag, 
                    y_hat_mean.squeeze() - 2*y_hat_std_diag, color = 'gray')
    plt.legend(['Training points', 
                'Predict function', 
                'True function', 
                'Confidence interval'])
    plt.show()
    pass