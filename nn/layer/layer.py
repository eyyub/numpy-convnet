from abc import ABCMeta, abstractmethod

class AbstractLayer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def train_forward(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def get_layer_error(self, z, backwarded_err):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, layer_err):
        raise NotImplementedError()

    @abstractmethod
    def get_grad(self, inputs, layer_err):
        raise NotImplementedError()

    @abstractmethod
    def update(self, grad):
        raise NotImplementedError()
