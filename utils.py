import numpy as np
import tensorflow as tf
import h5py

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

def get_dataname(volume_idx, slice_idxs):
    names = []
    for slice_idx in range(len(slice_idxs)+1):
        names.append("volume_" + str(volume_idx) + "_slice_" + str(slice_idx))
    return names

# TODO: Split the data into training and testing
def split_data(split_ratio):
    volume_idxs = np.arange(1, 370)
    split = int(len(volume_idxs)*split_ratio)
    train_idxs = volume_idxs[:split]
    test_idxs = volume_idxs[split:]
    slice_idxs = np.arange(0, 154)
    train_names = np.array([get_dataname(i, slice_idxs) for i in train_idxs]).flatten()
    test_names = np.array([get_dataname(i, slice_idxs) for i in test_idxs]).flatten()

    return train_names, test_names