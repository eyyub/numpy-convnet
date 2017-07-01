import numpy as np
from collections import deque

class Network(object):
    def __init__(self, layers, lr, loss):
        self.layers = layers
        self.loss = loss
        self._lr = lr

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, v):
        self._lr = v

    def forward(self, inputs):
        activation = inputs
        for l in self.layers:
            activation = l.forward(activation)
        return activation

    def train_step(self, mini_batch):
        mini_batch_inputs, mini_batch_outputs = mini_batch
        zs = deque([mini_batch_inputs])
        activation = mini_batch_inputs
        for l in self.layers:
            z, activation = l.train_forward(activation)
            zs.appendleft(z)

        loss_err = self.loss.deriv((activation, mini_batch_outputs))
        lz = zs.popleft()
        backwarded_err = loss_err
        grads = deque()
        for l in reversed(self.layers):
            layer_err = l.get_layer_error(lz, backwarded_err) #local
            lz = zs.popleft()
            grads.appendleft(l.get_grad(lz, layer_err))
            backwarded_err = l.backward(layer_err) # backwarded error

        # update step
        for l in self.layers:
            l.update(self.lr * grads.popleft())

        assert len(grads) == 0
