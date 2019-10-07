import read_mnist
import numpy as np

from datetime import datetime

class NeuralNetwork:
    def __init__(self,
                 nodes_on_hidden_layer=30, nodes_on_output_layer=10,
                 learning_rate_hidden=0.4, learning_rate_output=0.1):
        self.layers_size = [0, nodes_on_hidden_layer, nodes_on_output_layer]
        self.learning_rate_hidden = learning_rate_hidden
        self.learning_rate_output = learning_rate_output
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
        storage['W1'] = self.parameters['W1']
        storage['WX1'] = WX

        WX = self.parameters['W2'].dot(X)
        X = self.Softmax(WX)
        storage['X2'] = X
        storage['W2'] = self.parameters['W2']
        storage['WX2'] = WX

        return X, storage

    def calculate_derivatives(self, X, Y, storage):
        derivatives = {}
        storage['X0'] = X.T

        U = storage['X2']
        delta_j = U - Y.T
        dW = delta_j.dot(storage['X1'].T) / self.batch_size
        derivatives['dW2'] = dW
        delta_s = storage['W2'].T.dot(delta_j)

        dZ = delta_s * self.ReLU_derivative(storage['WX1'])
        dW = 1. / self.batch_size * dZ.dot(storage['X0'].T)
        derivatives['dW1'] = dW

        return derivatives

    def fit(self, X, Y, batch_size=128,  number_epochs=1000):
        np.random.seed(1)
        self.batch_size = batch_size
        self.n = X.shape[0]
        self.layers_size[0] = X.shape[1]

        self.initialize_parameters()
        for epoch in range(number_epochs):
            cost = 0.0
            X, Y = self.shuffle_arrays_together(X, Y)
            for i in range(0, self.n, self.batch_size):
                A, storage = self.fit_forward(X[i:i+self.batch_size])
                derivatives = self.calculate_derivatives(X[i:i+self.batch_size], Y[i:i+self.batch_size], storage)
                self.parameters['W1'] = self.parameters['W1'] - self.learning_rate_hidden * derivatives['dW1']
                self.parameters['W2'] = self.parameters['W2'] - self.learning_rate_output * derivatives['dW2']

                cost -= np.sum(Y[i:i+self.batch_size] * np.log(A.T))
            cost /= self.batch_size
            if epoch % 100 == 0:
                print("Cost: ", cost, "Train Accuracy:", self.predict(X, Y))

    def predict(self, X, Y):
        A = self.forward(X)
        y_hat = np.argmax(A, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (y_hat == Y).mean()
        return accuracy

    def shuffle_arrays_together(self, a, b):
        random_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(random_state)
        np.random.shuffle(b)
        return a, b


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = read_mnist.get_MNIST_Keras()
    train_x, test_x = read_mnist.normalized_MNIST(train_x, test_x)

    print("train_x shape: " + str(train_x.shape))
    print("test_x shape: " + str(test_x.shape))

    time_start = datetime.now()
    print('Time start: {}\n'.format(time_start))

    net = NeuralNetwork(nodes_on_hidden_layer=30,
                        learning_rate_hidden=0.1,
                        learning_rate_output=0.1)
    # 30, 0.4, 0.1

    net.fit(train_x, train_y, batch_size=128, number_epochs=20)
    print("Train Accuracy:", net.predict(train_x, train_y))
    print("Test Accuracy:", net.predict(test_x, test_y))

    time_end = datetime.now()
    print('\nTime end: {}\n'.format(time_end))
    print('Delta time =  {}\n'.format(time_end - time_start))

'''
30, 0.1, 0.1, 20
Train Accuracy: 0.9750833333333333
Test Accuracy: 0.965
Delta time =  0:00:13.629306
'''
