
from keras.models import Model
from keras.layers import Input, Dense

import read_mnist

batch_size = 60000 # in each iteration, we consider 128 training examples at once
num_epochs = 20 # we iterate twenty times over the entire training set
hidden_size = 30 # there will be 50 neurons in both hidden layers

height, width = 28, 28 # MNIST images are 28x28 and greyscale
num_classes = 10 # there are 10 classes (1 per digit)


(X_train, Y_train), (X_test, Y_test) = read_mnist.get_MNIST_Keras()
X_train, X_test = read_mnist.normalized_MNIST(X_train, X_test)

inp = Input(shape=(height * width,)) # Our input is a 1D vector of size 784
hidden_1 = Dense(hidden_size, activation='relu')(inp) # First hidden ReLU layer
out = Dense(num_classes, activation='softmax')(hidden_1) # Output softmax layer

model = Model(input=inp, output=out) # To define a model, just specify its input and output layers

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

model.fit(X_train, Y_train, # Train the model using the training set...
          batch_size=batch_size, nb_epoch=num_epochs,
          verbose=1)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])