import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import UNET
import h5py
import pathlib
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Hyperparameters
batch_size = 32
img_height = 240
img_width = 240

# Directory of training set
directory = r'C:\Users\Brody\Documents\College Material\Fall 2022\BIOE 485\FinalProject\BraTS2020_training_data\content\data\\'
ds_train = tf.data.Dataset.list_files(str(pathlib.Path(directory+'*.h5')))

@tf.function(input_signature=[tf.TensorSpec(None, tf.string)])
def process_path(file_path):
    # need to use non tf functions, call py_function
    image, mask = tf.py_function(func=process_path_helper, inp=[file_path], Tout=[tf.float32, tf.float32])
    image = tf.reshape(image, (240,240,4))
    mask = tf.reshape(mask, (240,240,3))
    return image, mask

def process_path_helper(file_path):
    # arbitrary Python code can be used here
    file_path = file_path.numpy().decode('ascii')
    f = h5py.File(file_path, 'r') 
    image = f['image'][()] 
    mask = f['mask'][()] 
    f.close()
    return image,mask

ds_train = ds_train.map(process_path).batch(batch_size)

inputs = keras.Input(shape=(240,240,4))
model = UNET.model(inputs)

if __name__ == "__main__":
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        metrics=["accuracy"]
    )
    model.summary()
    model.fit(ds_train, epochs=5)
    # Change to filepath of model to save
    model.save(r'C:\Users\Brody\Documents\College Material\Fall 2022\BIOE 485\FinalProject\models\unet_model.h5')

    # Change to directory of of test set
    directory = r'C:\Users\Brody\Documents\College Material\Fall 2022\BIOE 485\FinalProject\BraTS2020_training_data\content\test\\'
    ds_test = tf.data.Dataset.list_files(str(pathlib.Path(directory+'*.h5')))
    batch_size = 1
    ds_test = ds_test.map(process_path).batch(batch_size)
    model.evaluate(ds_test)