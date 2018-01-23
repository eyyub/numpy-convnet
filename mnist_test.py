import numpy as np
from nn.network import Network
from nn.layer.fullyconnected import FullyConnected
from nn.layer.flatten import Flatten
from nn.layer.conv import Conv
from nn.activation import sigmoid, relu , lkrelu , mse, linear, cross_entropy
import mnist_loader

def accuracy(net, X, Y):
    a = (np.argmax(cross_entropy._softmax(net.forward(X)), axis=1) == np.argmax(Y, axis=1))
    return np.sum(a) / float(X.shape[0]) * 100.

def one_hot(x, size):
    a = np.zeros((x.shape[0], size))
    a[np.arange(x.shape[0]), x] = 1.
    return a

if __name__ == '__main__':
    batch_size = 20

    # A simple strided convnet
    layers = [
        Conv((4, 4, 1, 20), strides=2, activation=lkrelu, filter_init=lambda shp: np.random.normal(size=shp) * np.sqrt(1.0 / (28*28 + 13*13*20)) ),
        Conv((5, 5, 20, 40), strides=2, activation=lkrelu, filter_init=lambda shp:  np.random.normal(size=shp) *  np.sqrt(1.0 / (13*13*20 + 5*5*40)) ),
        Flatten((5, 5, 40)),
        FullyConnected((5*5*40, 100), activation=sigmoid, weight_init=lambda shp: np.random.normal(size=shp) * np.sqrt(1.0 / (5*5*40 + 100.))),
        FullyConnected((100, 10), activation=linear, weight_init=lambda shp: np.random.normal(size=shp) * np.sqrt(1.0 / (110.)))
    ]
    lr = 0.001
    k = 2000
    net = Network(layers, lr=lr, loss=cross_entropy)

    (train_data_X, train_data_Y), v, (tx, ty) = mnist_loader.load_data('./data/mnist.pkl.gz')
    train_data_Y = one_hot(train_data_Y, size=10)
    ty = one_hot(ty, size=10)
    train_data_X = np.reshape(train_data_X, [-1, 28, 28, 1])
    tx = np.reshape(tx, [-1, 28, 28, 1])
    for epoch in xrange(100000):
        shuffled_index = np.random.permutation(train_data_X.shape[0])

        batch_train_X = train_data_X[shuffled_index[:batch_size]]
        batch_train_Y = train_data_Y[shuffled_index[:batch_size]]
        net.train_step((batch_train_X, batch_train_Y))

        loss = np.sum(cross_entropy.compute((net.forward(batch_train_X), batch_train_Y)))
        print 'Epoch: %d loss : %f' % (epoch, loss)
        if epoch % 1000 == 1:
            print 'Accuracy on first 500 test set\'s batch : %f' % accuracy(net, tx[:500], ty[:500])
        if epoch % 5000 == 5000 - 1:
            print 'Accuracy over all test set %f' % accuracy(net, tx, ty)
