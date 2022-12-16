import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import UNET
import utils
import pathlib
import argparse
import shutil
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser(description='Train UNET model.')
parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=int, default=.001, help='learning rate')
args = parser.parse_args()


# Hyperparameters
num_epochs = args.num_epochs
batch_size = args.batch_size
lr = args.lr
loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
print("<<<Hyperparameters initialized>>>")

# TODO: Change directory of training set
directory = r'/home/sl84/ECE_471_MP3/archive/BraTS2020_training_data/content/data//'

# TODO: Create test and train dataset directories and copy relevant data files to each directory
train_dir = '/home/sl84/ECE_471_MP3/train_data'
test_dir = '/home/sl84/ECE_471_MP3/test_data'
# shutil.rmtree(train_dir); shutil.rmtree(test_dir)
if not os.path.exists(train_dir) and not os.path.exists(test_dir):
    # TODO: Split the data into training and testing
    train_names, test_names = utils.split_data(0.8)
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    print("<<<Created training and testing dataset directories>>>")
    for train_name in tqdm(train_names):
        shutil.copy(directory+train_name+'.h5', train_dir)
    for test_name in tqdm(test_names):
        shutil.copy(directory+test_name+'.h5', test_dir)
    print("<<<Data split completed>>>")

ds_train = tf.data.Dataset.list_files(str(pathlib.Path(train_dir+"//*")))
ds_train = ds_train.map(utils.process_path).batch(batch_size)
print(ds_train)

inputs = keras.Input(shape=(240,240,4))
model = UNET.model(inputs)

if __name__ == "__main__":
    model.compile(
        loss=loss_func,
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=["accuracy"]
    )
    model.summary()
    model.fit(ds_train, epochs=num_epochs)
    # TODO: Change to filepath of model to save
    model.save(r'unet_model_final.h5')

    # TODO: Change to directory of of test set
    # directory = r'/archive/BraTS2020_training_data/content/test//'
    # ds_test = tf.data.Dataset.list_files(str(pathlib.Path(directory+'*.h5')))
    # batch_size = 1
    # ds_test = ds_test.map(utils.process_path).batch(batch_size)
    # model.evaluate(ds_test)