import numpy as np

class kernels:
    SQUARE_EXP = lambda x, y, **kwargs: np.exp(
        -np.dot((x - y), (x - y)) / (2 * kwargs['length'] ** 2))
    CONS = lambda x, y, **kwargs: kwargs['length']
    LINEAR = lambda x, y, **kwargs: np.dot(x, y)


class dumb_gaussian():
    def __init__(self, kernel, **kwargs):
        self.data = {'inputs': [], 'labels': []}
        self.kernel = lambda x, y: kernel(x, y, **kwargs)
        self.covar = np.array([[]])
        self.inv_covar = np.array([[]])

    def add_single_data(self, label, input, noise = 0.001):
        diag = self.kernel(input, input)+noise

        if self.covar.size > 0:
            new_row = np.array([self._gen_covar(input)])
            self.covar = np.concatenate((self.covar, new_row), axis=0)
            new_row = np.concatenate((new_row, np.array([[diag]])), axis=1)
            self.covar = np.concatenate((self.covar, new_row.T), axis=1)
        else:
            self.covar = np.matrix([[diag]])

        self.data['inputs'].append(input)
        self.data['labels'].append(label)
        self.inv_covar = np.array([[]])

    def _gen_covar(self, input):
        return np.array([
            self.kernel(input, x) for x in self.data['inputs']])

    def add_data(self, labels, inputs):
        assert len(labels) == len(
            inputs), 'Error -- different number of data and labels'
        for i, label in enumerate(labels):
            self.add_single_data(label, input[i])

    def evaluate(self, x):
        if self.inv_covar.size == 0:
            assert self.covar.size > 0, 'Error -- evaluation without any data'
            self.inv_covar = np.linalg.inv(self.covar)

        weights = np.dot(self._gen_covar(x), self.inv_covar)
        labels = np.array(self.data['labels'])
        return np.dot(weights, labels)
