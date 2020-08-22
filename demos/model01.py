# notebook source at: https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb

# import libs
import tensorflow as tf
import numpy as np
from tensorflow import keras

# define the neuronal network:
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# define an optimizer and a loss function
model.compile(optimizer='sgd', loss='mean_squared_error')

# define de input
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)


# training part
model.fit(xs, ys, epochs=500)

# predict the output for some value
print(model.predict([10.0]))
