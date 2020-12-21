import numpy as np
import tensorflow as tf
import os

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.datasets import mnist

from tensorflow.keras import Input, Model
from tensorflow.keras.utils import to_categorical

from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser

from keras.layers import ZeroPadding2D

path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolotemp'))

#model 
model_name = 'testmodel'
learn_new = True

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

np.savez_compressed(os.path.join(path_wd, 'x_test'), x_test)
np.savez_compressed(os.path.join(path_wd, 'y_test'), y_test)

np.savez_compressed(os.path.join(path_wd, 'x_norm'), x_train[::10])


#model


input_shape = x_train.shape[1:]

input_layer = Input(input_shape)

x = layers.Conv2D(filters=64,
               kernel_size=(3, 3),
               strides=(1, 1))(input_layer)

x = ZeroPadding2D(((1,0),(1,0)))(x)
x = ZeroPadding2D(((1,0),(1,0)))(x)



x = layers.Conv2D(filters=64,
               kernel_size=(3, 3),
               strides=(1, 1))(x)

x = layers.Flatten()(x)

x = layers.Dense(128, activation="relu")(x)


outputs = layers.Dense(10, activation="relu")(x)





model = keras.Model(inputs=input_layer, outputs=outputs, name="mnist_model")

model.compile('adam', 'categorical_crossentropy', ['accuracy'])

# Train model with backprop.
model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=2,
          validation_data=(x_test, y_test))



model.save((os.path.join(path_wd, model_name + '.h5')))

configparser = import_configparser()
config = configparser.ConfigParser()



config['paths'] = {
    'path_wd': path_wd,             # Path to model.
    'dataset_path': path_wd,        # Path to dataset.
    'filename_ann': model_name      # Name of input model.
}

config['tools'] = {
    'evaluate_ann': True,           # Test ANN on dataset before conversion.
    'normalize': True               # Normalize weights for full dynamic range.
}

config['simulation'] = {
    'simulator': 'INI',             # Chooses execution backend of SNN toolbox.
    'duration': 50,                 # Number of time steps to run each sample.
    'num_to_test': 100,             # How many test samples to run.
    'batch_size': 50,               # Batch size for simulation.
    'keras_backend': 'tensorflow'   # Which keras backend to use.
}



config_filepath = os.path.join(path_wd, 'config')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)

main(config_filepath)

