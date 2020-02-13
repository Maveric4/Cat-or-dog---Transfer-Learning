import os
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
import dg_utils.dg_utils as dg_utils
img_size = 200

# Define a Callback class that stops training once accuracy reaches 99.9%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.99):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True

## Load pretrained model
local_weights_file = os.path.dirname(__file__) + '/checkpoints/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=(img_size, img_size, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed9')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

## Add top layers
from tensorflow.keras.optimizers import RMSprop

# try this one:
# In my case, I set input image size for the InceptionV3 model to img_size x img_size pixels, and my mixed9 layer ended up having the shape of 8 x 8 x 2048 = 131,072. I've slapped 3 1024-neuron dense layers on top followed by 30% dropout each, and was able to hit 100% accuracy after 35 epochs:

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
# x = layers.Dense(512, activation='relu')(x)
# # Add a dropout rate of 0.2
# x = layers.Dropout(0.2)(x)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.3
x = layers.Dropout(0.3)(x)
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.3
x = layers.Dropout(0.3)(x)
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.3
x = layers.Dropout(0.3)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['acc'])

## Get training and validation datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Add our data-augmentation parameters to ImageDataGenerator
train_dir = dg_utils.get_train_data_path(__file__)
validation_dir = dg_utils.get_valid_data_path(__file__)

train_datagen_augmented = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

# Note that the validation data should not be augmented!
valid_datagen_augmented = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen_augmented.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(img_size, img_size))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = valid_datagen_augmented.flow_from_directory(validation_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(img_size, img_size))

    ## Training model
print("--------------------------------------------------------------------------------")
answer = input("Write 1 if you want to train again neural network or 2 if you want to load model")
callbacks = myCallback()
if int(answer) == 1:
    tic = dg_utils.start_training_time_measurement(__file__)
    history = model.fit_generator(
                train_generator,
                validation_data=validation_generator,
                steps_per_epoch=100,
                epochs=80,
                validation_steps=50,
                verbose=2,
                callbacks=[callbacks])
    dg_utils.end_training_time_measurement(__file__, tic)

    ## Saving model
    dg_utils.save_model(__file__, model, version=5)

    ## Plot accuracy and loss

    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure()
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.show()
## Loading model
else:
    model = dg_utils.load_model(__file__, version=5)

## Checking accuracy of model
TESTING_DOGS_DIR = dg_utils.get_test_data_path(__file__) + "/dogs/"
TESTING_CATS_DIR = dg_utils.get_test_data_path(__file__) + "/cats/"
zuzq_testing = os.path.dirname(__file__) + "/data/Zuzq_testing/"
locations = [TESTING_DOGS_DIR, TESTING_CATS_DIR, zuzq_testing]
from tensorflow.keras.preprocessing import image
for loc in locations:
    test_names = random.sample(os.listdir(loc), min(10, len(os.listdir(loc))))
    for image_name in test_names:
        try:
            path = loc + image_name
            # img = tf.keras.preprocessing.image.load_img(os.path.normpath(path), target_size=(img_size, img_size))
            img = image.load_img(os.path.normpath(path), target_size=(img_size, img_size))
        except:
            print("Something went wrong!")
            pass
        x = image.img_to_array(img) / 255.
        x = np.expand_dims(x, axis=0)
        x = np.vstack([x])
        prediction = model.predict(x)
        print(prediction)
        if prediction > 0.5:
            print(path.split('/')[-2] + '/' + image_name + " is a dog")
        else:
            print(path.split('/')[-2] + '/' + image_name + " is a cat")


##

