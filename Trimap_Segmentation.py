import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import nibabel as nib
import pathlib
import h5py
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)

# Read data
# train_metad = pd.read_csv('Data/BraTS20 Training Metadata.csv')

# Data parser
datadir = '/Users/sean/Documents/GitHub/Trimap-Segmentation/Data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'

def get_data(filetype):
  images = []
  for subdir, dirs, files in os.walk(datadir):
      for file in files:
        if filetype in file:
          filename = os.path.join(subdir, file)
          try:
            img = nib.load(filename)
            img = np.asanyarray(img.dataobj)
            img = np.rot90(img)
            images.append(img)
          except:
            continue
        else:
          continue
  return images

# Plot helper function
def plot(images):
  l = len(images)
  rows = cols = (l//2) + 1
  fig, ax = plt.subplots(1, l)
  for i in range(l):
    ax[i].imshow(images[i][:,:,65], cmap='bone')
    ax[i].axis('off')
  plt.show()

images = get_data('flair')
masks = get_data('seg')
# Plot sample images
plot(images[:4])
plot(masks[:4])

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

# print(model.summary())

# chose Adam based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7407771/
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# history = model.fit(train_images, train_labels, epochs=10, 
#                     validation_data=(test_images, test_labels))

# Initialize hyperparameters
batch_size = 1
img_height = 240
img_width = 240