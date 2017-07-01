from abc import ABCMeta, abstractmethod
import numpy as np

class AbstractActivation(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def compute(self, x):
        raise NotImplementedError()

    @abstractmethod
    def deriv(self, x):
        raise NotImplementedError()

class Relu(AbstractActivation):
    def compute(self, x):
        return np.maximum(0, x)

    def deriv(self, x):
        return 1. * (x > 0)

class LeakyRelu(AbstractActivation):
    def compute(self, x):
        return np.maximum(0.01, x)

    def deriv(self, x):
        g = 1. * (x > 0)
        g[g == 0.] = 0.01
        return g

class Sigmoid(AbstractActivation):
    def compute(self, x):
        return 1. / (1. + np.exp(-x))

    def deriv(self, x):
        y = self.compute(x)
        return y * (1. - y)

class Linear(AbstractActivation):
    def compute(self, x):
        return x

    def deriv(self, x):
        return 1.

class Loss(AbstractActivation):
    pass

class MeanSquaredError(Loss):
    def compute(self, (X, Y)):
        return (1. / 2. * X.shape[0]) * ((X - Y) ** 2.)

    def deriv(self, (X, Y)):
        return (X - Y) / X.shape[0]

class CrossEntropy(Loss):
    def _softmax(self, X):
        expvx = np.exp(X - np.max(X, axis=1)[..., np.newaxis])
        return expvx/np.sum(expvx, axis=1, keepdims=True)

    def compute(self, (X, Y)):
        sf = self._softmax(X)
        return -np.log(sf[np.arange(X.shape[0]), np.argmax(Y, axis=1)]) / X.shape[0]

    def deriv(self, (X, Y)):
        err = self._softmax(X)
        return (err - Y) / X.shape[0]

relu = Relu()
lkrelu = LeakyRelu()
sigmoid = Sigmoid()
linear = Linear()

mse = MeanSquaredError()
cross_entropy = CrossEntropy()
