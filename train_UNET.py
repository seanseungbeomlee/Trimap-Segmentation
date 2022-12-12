import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import UNET
import utils
import h5py
import pathlib
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Hyperparameters
batch_size = 32
img_height = 240
img_width = 240

# TODO: Change directory of training set
directory = r'/home/sl84/ECE_471_MP3/archive/BraTS2020_training_data/content/data//'
ds_train = tf.data.Dataset.list_files(str(pathlib.Path(directory+'*.h5')))

ds_train = ds_train.map(utils.process_path).batch(batch_size)

inputs = keras.Input(shape=(240,240,4))
model = UNET.model(inputs)

if __name__ == "__main__":
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        metrics=["accuracy"]
    )
    model.summary()
    model.fit(ds_train, epochs=10)
    # TODO: Change to filepath of model to save
    model.save(r'/models/unet_model.h5')

    # TODO: Change to directory of of test set
    # directory = r'/archive/BraTS2020_training_data/content/test//'
    # ds_test = tf.data.Dataset.list_files(str(pathlib.Path(directory+'*.h5')))
    # batch_size = 1
    # ds_test = ds_test.map(utils.process_path).batch(batch_size)
    # model.evaluate(ds_test)