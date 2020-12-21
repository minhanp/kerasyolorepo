import numpy as np
import tensorflow as tf
import os
from tensorflow import keras

from tensorflow.keras import Input, Model
from tensorflow.keras.utils import to_categorical

from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser

from tensorflow import keras



path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolotemp'))

model_name = 'yolomodel'

model = keras.models.load_model((os.path.join(path_wd, model_name + '.h5')))


configparser = import_configparser()
config = configparser.ConfigParser()



config['paths'] = {
    'path_wd': path_wd,             # Path to model.
    'dataset_path': path_wd,        # Path to dataset.
    'filename_ann': model_name      # Name of input model.
}

config['tools'] = {
    'evaluate_ann': True,           # Test ANN on dataset before conversion.
    'simulate': False,
    'normalize': True               # Normalize weights for full dynamic range.
}

config['simulation'] = {
    'simulator': 'INI',             # Chooses execution backend of SNN toolbox.
    'duration': 1,                 # Number of time steps to run each sample.
    'num_to_test': 1,             # How many test samples to run.
    'batch_size': 1,               # Batch size for simulation.
    'keras_backend': 'tensorflow'   # Which keras backend to use.
}



config_filepath = os.path.join(path_wd, 'config')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)

main(config_filepath)