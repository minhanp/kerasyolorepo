import numpy as np
import tensorflow as tf
import os

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input, Model
from tensorflow.keras.datasets import mnist

path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temp'))

model_name = 'keras_lol1_INI'


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


x_test = x_test[..., tf.newaxis].astype("float32")


y_test = to_categorical(y_test, 10)


model = keras.models.load_model((os.path.join(path_wd, model_name + '.h5')))

results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)