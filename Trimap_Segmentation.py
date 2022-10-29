import tensorflow as tf
import tensorflow_datasets as tfds

df = tfds.load('oxford_iiit_pet', split='train', shuffle_files=True)
df = df.shuffle(1024).batch(64).prefetch(tf.data.AUTOTUNE)
for example in df.take(1):
  image, label = example["image"], example["label"]