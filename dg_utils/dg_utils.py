import tensorflow as tf
import numpy as np
import os
import time
import datetime


## Check whether checkpoints directory exists
def checkpoint_dir_check(file_path):
    checkpoint_dir = os.path.dirname(file_path) + "/checkpoints/"
    if not os.path.isdir(checkpoint_dir):
        print("No checkpoints folder!")
        os.mkdir(checkpoint_dir)


def save_weights(file_path, model):
    checkpoint_dir_check(file_path)
    model.save_weights(os.path.dirname(file_path) + "/checkpoints/my_checkpoint")


def load_weights(file_path, model):
    checkpoint_dir_check(file_path)
    model.load_weights(os.path.dirname(file_path) + "/checkpoints/my_checkpoint")
    return model


def save_model(file_path, model, version=0):
    """
    This technique saves:
        The weight values
        The model's configuration(architecture)
        The optimizer configuration
    """
    checkpoint_dir_check(file_path)
    model.summary()
    print("Model saved")
    with open(os.path.dirname(file_path) + "/checkpoints/time.txt", mode='a', encoding = 'utf-8') as f:
        print("Timestamp: {}. {}my_model saved.".format(datetime.datetime.now(), version), file=f)
    model.save(os.path.dirname(file_path) + "/checkpoints/" + str(version) + 'my_model.h5')


def load_model(file_path, version=-1):
    """
    This technique loades:
        The weight values
        The model's configuration(architecture)
        The optimizer configuration
    """
    checkpoint_dir_check(file_path)
    if version != -1:
        new_model = tf.keras.models.load_model(os.path.dirname(file_path) + "/checkpoints/" + str(version) + 'my_model.h5')
    else:
        print("Looking for version")
        for version in range(10, 0, -1):
            try:
                new_model = tf.keras.models.load_model(os.path.dirname(file_path) + "/checkpoints/" + str(version) + 'my_model.h5')
                break
            except:
                pass
    new_model.summary()
    print("Model loaded")
    return new_model


def load_latest_model(file_path, model):
    checkpoint_dir_check(file_path)
    latest_checkpoint = tf.train.latest_checkpoint(os.path.dirname(file_path) + "/checkpoints/checkpoint.ckpt")
    model.load_weights(latest_checkpoint)
    return model
# # Create checkpoint callback
# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     checkpoint_dir, verbose=1, save_weights_only=True,
#     # Save weights, every 5-epochs.
#     period=5)


def get_valid_data_path(file_path):
    folders = ["/data/valid/", "/data/validation/"]
    for folder in folders:
        if os.path.isdir(os.path.dirname(file_path) + folder):
            validation_data_path = os.path.dirname(file_path) + folder
    return validation_data_path


def get_train_data_path(file_path):
    folders = ["/data/train/", "/data/training/"]
    for folder in folders:
        if os.path.isdir(os.path.dirname(file_path) + folder):
            train_data_path = os.path.dirname(file_path) + folder
    return train_data_path


def get_test_data_path(file_path):
    folders = ["/data/test/", "/data/testing/"]
    for folder in folders:
        if os.path.isdir(os.path.dirname(file_path) + folder):
            test_data_path = os.path.dirname(file_path) + folder
    return test_data_path

def start_training_time_measurement(file_path):
    checkpoint_dir_check(file_path)
    tic = time.time()
    return tic

def end_training_time_measurement(file_path, tic):
    training_time = time.time() - tic
    with open(os.path.dirname(file_path) + "/checkpoints/time.txt", mode='a', encoding='utf-8') as f:
        print("Timestamp: {}. It took: {} s or {} min.".format(datetime.datetime.now(), training_time, int(training_time/60)), file=f)
    print("Timestamp: {}. It took: {} s or {} min.".format(datetime.datetime.now(), training_time, int(training_time/60)))
    return training_time






