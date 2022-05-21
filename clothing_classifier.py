import tensorflow as tf
import numpy as np
from tensorflow import keras
import random

# for displaying img
import matplotlib.pyplot as plt

rand_img = random.randint(0, 10000)

# load a predefined dataset
fm = keras.datasets.fashion_mnist

# pull out data from dataset
(train_imgs, train_labels), (test_imgs, test_labels) = fm.load_data()

# defining neural net structure
model = keras.Sequential([

    # input is a 28x28 img ("Flatten" flattens the 28x28 into a single 784x1 input layer)
    keras.layers.Flatten(input_shape=(28, 28)),

    # hidden layer with 128 deep
    keras.layers.Dense(units=128, activation=tf.nn.relu),

    # output is 0-10 (depending on clothing)
    keras.layers.Dense(units=10, activation=tf.nn.softmax)

])

# compile the model
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# training the model
model.fit(train_imgs, train_labels, epochs=5)

# make prediction
predictions = model.predict(test_imgs)

# testing the model using our testing data
test_loss = model.evaluate(test_imgs, test_labels)

print("Test loss: ", test_loss[0])
print("Accuracy: ", test_loss[1])

# print out prediction
print("predicted label : ", list(predictions[rand_img]).index(max(predictions[rand_img])))

# print the correct label
print("Correct label : ", test_labels[rand_img])

# for showing the img
plt.imshow(test_imgs[rand_img])
plt.show()
