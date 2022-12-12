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