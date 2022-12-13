import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import utils
import argparse
import pathlib
import h5py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser(description='Train UNET model.')
parser.add_argument('--model', help='select model')
args = parser.parse_args()

# TODO: Load model
directory = r'/home/sl84/ECE_471_MP3/'
model_name = args.model
model = load_model(directory + model_name)

# TODO: Load test dataset
directory = r'/home/sl84/ECE_471_MP3/test_data/'
ds_test = tf.data.Dataset.list_files(str(pathlib.Path(directory+'*.h5')))
batch_size = 1
ds_test = ds_test.map(utils.process_path).batch(batch_size)

num_tumor = 0
num_good = 0
num_tumor_correct = 0
num_good_correct = 0
num_false_tumor = 0
num_false_good = 0

count = 0
for batch in ds_test:
    if count % 500 == 0:
        print(count)

    image, mask = batch
    y = model(image).numpy()
    mask = mask.numpy()

    zeros = y < -5
    ones = y >= -5
    y[zeros] = 0.0
    y[ones] = 1.0
    
    num_tumor += np.sum(mask)
    num_good += np.sum(mask == 0.0)

    num_tumor_correct += np.sum(np.logical_and(y == 1.0, mask == 1.0))
    num_good_correct += np.sum(np.logical_and(y == 0.0, mask == 0.0))
    num_false_tumor += np.sum(np.logical_and(y == 1.0, mask == 0.0))
    num_false_good += np.sum(np.logical_and(y == 0.0, mask == 1.0))
        
    count += 1

# Print model metrics
print(model_name)
print('PPV: ', str(num_tumor_correct/num_tumor))
print('NPV: ', str(num_good_correct/num_good))
print('Sensitivity: ', str(num_tumor_correct/(num_tumor_correct + num_false_good)))
print('Specificity: ', str(num_good_correct/(num_good_correct + num_false_tumor)))

# Loss curve
loss = np.array([0.7215,0.6941,0.6934,0.6932,0.6932,0.6932,0.6932,0.6932,0.6931,0.6931])
epochs = range(1,11)
plt.figure()
plt.plot(epochs,loss)
plt.title('Loss vs Number of Epochs')
plt.xlabel('Number of Epochs')
plt.ylabel('Binary Cross Entropy Loss')