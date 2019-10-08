import read_mnist
import numpy as np

from datetime import datetime


class NeuralNetwork:
    def __init__(self, nodes_hidden=30, nodes_output=10, lr_hidden=0.4, lr_output=0.1):
        self.layers_size = [0, nodes_hidden, nodes_output]
        self.lr_hidden = lr_hidden
        self.lr_output = lr_output
        self.parameters = {}
        self.n = 0
        self.batch_size = 0

    def ReLU(self, X):
        return X * (X > 0)

    def ReLU_derivative(self, X):
        return 1. * (X > 0)

    def Softmax(self, X):
        expX = np.exp(X)
        return expX / expX.sum(axis=0, keepdims=True)

    def initialize_parameters(self):
        self.parameters['W1'] = 2 * np.random.randn(self.layers_size[1], self.layers_size[0]) / \
                                np.sqrt(self.layers_size[0])
        self.parameters['W2'] = 2 * np.random.randn(self.layers_size[2], self.layers_size[1]) / \
                                np.sqrt(self.layers_size[1])

    def forward(self, X):
        X = X.T
        WX = self.parameters['W1'].dot(X)
        X = self.ReLU(WX)

        WX = self.parameters['W2'].dot(X)
        X = self.Softmax(WX)

        return X

    def fit_forward(self, X):
        storage = {}

        X = X.T
        WX = self.parameters['W1'].dot(X)
        X = self.ReLU(WX)
        storage['X1'] = X
        storage['WX1'] = WX

        WX = self.parameters['W2'].dot(X)
        X = self.Softmax(WX)
        storage['X2'] = X
        storage['W2'] = self.parameters['W2']

        return X, storage

    def calculate_derivatives(self, X0, Y, storage):  # X0 - вход на сеть
        delta_2 = Y.T - storage['X2']  # Y - U
        dW2 = delta_2.dot(storage['X1'].T) / self.batch_size

        delta_1 = storage['W2'].T.dot(delta_2) * self.ReLU_derivative(storage['WX1'])
        dW1 = delta_1.dot(X0) / self.batch_size

        return dW1, dW2

    def fit(self, X, Y, batch_size=128, number_epochs=1000):
        np.random.seed(1)
        self.batch_size = batch_size
        self.n = X.shape[0]
        self.layers_size[0] = X.shape[1]
        self.initialize_parameters()

        epoch_percent = 0
        for epoch in range(number_epochs):
            cost = 0.0
            X, Y = self.shuffle_arrays_together(X, Y)
            for i in range(0, self.n, self.batch_size):
                U, storage = self.fit_forward(X[i:i + self.batch_size])
                dW1, dW2 = self.calculate_derivatives(X[i:i + self.batch_size], Y[i:i + self.batch_size], storage)
                self.parameters['W1'] = self.parameters['W1'] + self.lr_hidden * dW1
                self.parameters['W2'] = self.parameters['W2'] + self.lr_output * dW2
            if epoch % round(number_epochs / 10) == 0:
                epoch_percent += 10
                accuracy, crossentropy = self.predict(X, Y)
                print('{}% Train Accuracy: {}; Train loss: {}'.format(epoch_percent, accuracy, crossentropy))

    def predict(self, X, Y):
        U = self.forward(X)
        crossentropy = -np.sum(Y * np.log(U.T)) / X.shape[0]

        U = np.argmax(U, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (U == Y).mean()
        return crossentropy, accuracy

    def shuffle_arrays_together(self, a, b):
        random_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(random_state)
        np.random.shuffle(b)
        return a, b


def fit_and_test_net_on_MNIST(hidden_size=30, batch_size=128, num_epochs=20, lr_hidden=0.1, lr_output=0.1):
    (train_x, train_y), (test_x, test_y) = read_mnist.get_MNIST_Keras()
    train_x, test_x = read_mnist.normalized_MNIST(train_x, test_x)

    time_start = datetime.now()
    net = NeuralNetwork(nodes_hidden=hidden_size, lr_hidden=lr_hidden, lr_output=lr_output)
    net.fit(train_x, train_y, batch_size=batch_size, number_epochs=num_epochs)

    delta_time = datetime.now() - time_start
    score_train = net.predict(train_x, train_y)
    score_test = net.predict(test_x, test_y)
    return score_train, score_test, delta_time


if __name__ == '__main__':
    score_train, score_test, delta_time = fit_and_test_net_on_MNIST(30, 128, 20, 0.1, 0.1)
    print()
    print('Delta time =', delta_time)
    print('Test loss:', score_test[0])
    print('Test accuracy:', score_test[1])


'''
30, 0.1, 0.1, 20
Train Accuracy: 0.9750833333333333
Test Accuracy: 0.965
Delta time =  0:00:13.629306
'''
'''
на 256 нейронов
Train Accuracy: 0.9915166666666667
Test Accuracy: 0.9772

Delta time =  0:01:01.758172
'''
