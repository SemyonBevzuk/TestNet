import read_mnist
import numpy as np

from datetime import datetime


class NeuralNetwork:
    def __init__(self, nodes_hidden=30, nodes_output=10, lr=0.1):
        self._layers_size = [0, nodes_hidden, nodes_output]
        self._lr = lr
        self._W1 = np.array([])
        self._W2 = np.array([])
        self._b1 = np.array([])
        self._b2 = np.array([])
        self._n = 0
        self._batch_size = 0

    def forward(self, X):
        X = X.T
        WX = self._W1.dot(X) + self._b1
        X = self._ReLU(WX)

        WX = self._W2.dot(X) + self._b2
        X = self._Softmax(WX)

        return X

    def predict(self, X, Y):
        U = self.forward(X)
        crossentropy = -np.sum(Y * np.log(U.T)) / X.shape[0]

        U = np.argmax(U, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (U == Y).mean()

        return crossentropy, accuracy

    def fit(self, X, Y, batch_size=128, number_epochs=10):
        np.random.seed(1)
        self._batch_size = batch_size
        self._n = X.shape[0]
        self._layers_size[0] = X.shape[1]
        self._initialize_weights()

        epoch_percent = 0
        for epoch in range(number_epochs):
            X, Y = self._shuffle_arrays_together(X, Y)
            for i in range(0, self._n, self._batch_size):
                U, storage = self._fit_forward(X[i:i + self._batch_size])
                dW1, dW2, db1, db2 = self._calculate_derivatives(X[i:i + self._batch_size], Y[i:i + self._batch_size], U, storage)
                self._W1 = self._W1 + self._lr * dW1
                self._W2 = self._W2 + self._lr * dW2
                self._b1 = self._b1 + self._lr * db1
                self._b2 = self._b2 + self._lr * db2

            if epoch % round(number_epochs / 10) == 0:
                epoch_percent += 10
                crossentropy, accuracy = self.predict(X, Y)
                print('{}% Train Accuracy: {}; Train loss: {}'.format(epoch_percent, accuracy, crossentropy))

    def _initialize_weights(self):
        sigma_1 = 2. / np.sqrt(self._layers_size[0] + self._layers_size[1])
        sigma_2 = 2. / np.sqrt(self._layers_size[1] + self._layers_size[2])
        self._W1 = sigma_1 * np.random.randn(self._layers_size[1], self._layers_size[0])
        self._W2 = sigma_2 * np.random.randn(self._layers_size[2], self._layers_size[1])
        self._b1 = np.zeros((self._layers_size[1], 1))
        self._b2 = np.zeros((self._layers_size[2], 1))

    def _fit_forward(self, X):
        storage = {}

        X = X.T
        WX = self._W1.dot(X) + self._b1
        X = self._ReLU(WX)
        storage['V'] = X
        storage['WX1'] = WX

        WX = self._W2.dot(X) + self._b2
        X = self._Softmax(WX)

        return X, storage

    def _calculate_derivatives(self, X0, Y, U, storage):
        delta_2 = Y.T - U
        dW2 = delta_2.dot(storage['V'].T) / self._batch_size
        db2 = np.sum(delta_2, axis=1, keepdims=True) / self._batch_size

        delta_1 = self._W2.T.dot(delta_2) * self._ReLU_derivative(storage['WX1'])
        dW1 = delta_1.dot(X0) / self._batch_size
        db1 = np.sum(delta_1, axis=1, keepdims=True) / self._batch_size

        return dW1, dW2, db1, db2

    @staticmethod
    def _ReLU(X):
        return X * (X > 0)

    @staticmethod
    def _ReLU_derivative(X):
        return 1. * (X > 0)

    @staticmethod
    def _Softmax(X):
        expX = np.exp(X)
        return expX / expX.sum(axis=0, keepdims=True)

    @staticmethod
    def _shuffle_arrays_together(a, b):
        random_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(random_state)
        np.random.shuffle(b)
        return a, b


def fit_and_test_net_on_MNIST(hidden_size=30, batch_size=128, num_epochs=20, lr=0.1):
    (train_x, train_y), (test_x, test_y) = read_mnist.get_MNIST_Keras()
    train_x, test_x = read_mnist.normalized_MNIST(train_x, test_x)

    time_start = datetime.now()
    net = NeuralNetwork(nodes_hidden=hidden_size, lr=lr)
    net.fit(train_x, train_y, batch_size=batch_size, number_epochs=num_epochs)

    delta_time = datetime.now() - time_start
    score_train = net.predict(train_x, train_y)
    score_test = net.predict(test_x, test_y)
    return score_train, score_test, delta_time


if __name__ == '__main__':
    score_train, score_test, delta_time = fit_and_test_net_on_MNIST(20, 128, 20, 0.1)
    print()
    print('Delta time =', delta_time)
    print('Test loss:', score_test[0])
    print('Test accuracy:', score_test[1])

