from tensorflow import keras
from tensorflow.keras import layers

def model(inputs):
    # model
    l1_in = layers.BatchNormalization()(inputs)
    
    # Level 1
    x1 = layers.Conv2D(filters=32,kernel_size=3,padding='same')(l1_in)
    x1 = keras.activations.relu(x1)
    
    xx1 = layers.Conv2D(filters=32,kernel_size=3,padding='same')(x1)
    xx1 = keras.activations.relu(xx1)
    
    # Level 2
    l2_in = layers.MaxPool2D()(xx1)
    
    x2 = layers.Conv2D(filters=64,kernel_size=3,padding='same')(l2_in)
    x2 = keras.activations.relu(x2)
    
    xx2 = layers.Conv2D(filters=64,kernel_size=3,padding='same')(x2)
    xx2 = keras.activations.relu(xx2)
    
    # Level 3
    l3_in = layers.MaxPool2D()(xx2)
    
    x3 = layers.Conv2D(filters=128,kernel_size=3,padding='same')(l3_in)
    x3 = keras.activations.relu(x3)
    
    xx3 = layers.Conv2D(filters=128,kernel_size=3,padding='same')(x3)
    xx3 = keras.activations.relu(xx3)
    
    # Level 4
    l4_in = layers.MaxPool2D()(xx3)
    
    x4 = layers.Conv2D(filters=256,kernel_size=3,padding='same')(l4_in)
    x4 = keras.activations.relu(x4)
    
    xx4 = layers.Conv2D(filters=256,kernel_size=3,padding='same')(x4)
    xx4 = keras.activations.relu(xx4)
    
    # Level 5
    l5_in = layers.MaxPool2D()(xx4)
    
    x5 = layers.Conv2D(filters=512,kernel_size=3,padding='same')(l5_in)
    x5 = keras.activations.relu(x5)
    
    xx5 = layers.Conv2D(filters=512,kernel_size=3,padding='same')(x5)
    xx5 = keras.activations.relu(xx5)
    
    l5_out = layers.Conv2DTranspose(filters=256,kernel_size=3,padding='same')(xx5)
    l5_out = layers.UpSampling2D()(l5_out)
    
    # Level 4
    y4 = layers.Concatenate()([xx4, l5_out])
    
    yy4 = layers.Conv2D(filters=256,kernel_size=3,padding='same')(y4)
    yy4 = keras.activations.relu(yy4)
    
    yyy4 = layers.Conv2D(filters=256,kernel_size=3,padding='same')(yy4)
    yyy4 = keras.activations.relu(yyy4)
    
    l4_out = layers.Conv2DTranspose(filters=128,kernel_size=3,padding='same')(yyy4)
    l4_out = layers.UpSampling2D()(l4_out)
    
    # Level 3
    y3 = layers.Concatenate()([xx3, l4_out])
    
    yy3 = layers.Conv2D(filters=128,kernel_size=3,padding='same')(y3)
    yy3 = keras.activations.relu(yy3)
    
    yyy3 = layers.Conv2D(filters=128,kernel_size=3,padding='same')(yy3)
    yyy3 = keras.activations.relu(yyy3)
    
    l3_out = layers.Conv2DTranspose(filters=64,kernel_size=3,padding='same')(yyy3)
    l3_out = layers.UpSampling2D()(l3_out)
    
    # Level 2
    y2 = layers.Concatenate()([xx2, l3_out])
    
    yy2 = layers.Conv2D(filters=64,kernel_size=3,padding='same')(y2)
    yy2 = keras.activations.relu(yy2)
    
    yyy2 = layers.Conv2D(filters=64,kernel_size=3,padding='same')(yy2)
    yyy2 = keras.activations.relu(yyy2)
    
    l2_out = layers.Conv2DTranspose(filters=32,kernel_size=3,padding='same')(yyy2)
    l2_out = layers.UpSampling2D()(l2_out)
    
    # Level 1
    y1 = layers.Concatenate()([xx1, l2_out])
    
    yy1 = layers.Conv2D(filters=32,kernel_size=3,padding='same')(y1)
    yy1 = keras.activations.relu(yy1)
    
    yyy1 = layers.Conv2D(filters=32,kernel_size=3,padding='same')(yy1)
    yyy1 = keras.activations.relu(yyy1)
    
    outputs = layers.Conv2D(filters=3,kernel_size=1,padding='same')(yyy1)
    
    model = keras.Model(inputs=inputs, outputs = outputs)
    return model