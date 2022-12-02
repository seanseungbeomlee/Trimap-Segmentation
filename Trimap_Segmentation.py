import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import pathlib
import h5py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Read data
train_metad = pd.read_csv('Data/BraTS20 Training Metadata.csv')
# surv_info = pd.read_csv('Data/BraTS2020_training_data/content/data/survival_info.csv')
# name_map = pd.read_csv('Data/BraTS2020_training_data/content/data/name_mapping.csv')
# metad = pd.read_csv('Data/BraTS2020_training_data/content/data/meta_data.csv')
# print(train_metad)

directory = r'Data/BraTS2020_training_data/content/data//'
ds_train = tf.data.Dataset.list_files(str(pathlib.Path(directory+'*.h5')))

@tf.function(input_signature=[tf.TensorSpec(None, tf.string)])
def process_path(file_path):
    # need to use non tf functions, call py_function
    image, mask = tf.py_function(func=process_path_helper, inp=[file_path], Tout=[tf.float32, tf.float32])
    return image, mask

def process_path_helper(file_path):
    # arbitrary Python code can be used here
    file_path = file_path.numpy().decode('ascii')
    f = h5py.File(file_path, 'r') 
    image = f['image'][()] 
    mask = f['mask'][()] 
    f.close()
    return image,mask

# e.g. ds_train[0] gives pair of image and mask tensors
# can also call a batch() function on this
# ds_train = ds_train.map(process_path)
# print(ds_train.as_numpy_iterator().next())

x = np.array(h5py.File('Data/BraTS2020_training_data/content/data/volume_60_slice_6.h5', 'r'))
print(x)

# Initialize Model:
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))

print(model.summary())

# chose Adam based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7407771/
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# history = model.fit(train_images, train_labels, epochs=10, 
#                     validation_data=(test_images, test_labels))

batch_size = 1
img_height = 240
img_width = 240