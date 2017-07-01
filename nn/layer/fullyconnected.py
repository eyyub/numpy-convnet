from layer import AbstractLayer

class FullyConnected(AbstractLayer):
    def __init__(self, wshape, activation, weight_init):
        self.wshape = wshape
        self.W = weight_init(self.wshape)
        self.activation = activation

    def forward(self, inputs):
        return self.activation.compute(inputs.dot(self.W))

    def train_forward(self, inputs):
        z = inputs.dot(self.W)
        return (z, self.activation.compute(z))

    def get_layer_error(self, z, backwarded_err):
        return backwarded_err * self.activation.deriv(z)

    def backward(self, layer_err):
        return layer_err.dot(self.W.T)

    def get_grad(self, inputs, layer_err):
        return inputs.T.dot(layer_err)

    def update(self, grad):
        self.W -= grad
