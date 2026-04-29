import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from testing import TestingScript

import pandas as pd
import numpy as np

import sys



## EXPERIMENT WITH THESE:
batch_size = 64
img_height = 244
img_width = 244


##SPLIT DATA INTO TRAIN AND TEST

#TODO: starting with the positive negative dataset only
#TODO: use positive dataset, make work for positive classification
###### - ACTUALLY THIS IS LITERALLY A DATASET OF FIGURES... USE TBDB and Bacterium Quantity

#TODO: Add transformation images

    
# train_ds = tf.keras.utils.image_dataset_from_directory(
#     "Modified_Training_Data/TBDB",
#     validation_split=0.2,
#     subset="training",
#     seed=123,
#     image_size=(img_height, img_width),
#     batch_size=batch_size,
#     crop_to_aspect_ratio=True)
    


# val_ds = tf.keras.utils.image_dataset_from_directory(
#     "Modified_Training_Data/TBDB",
#     validation_split=0.2,
#     subset="validation",
#     seed=123,
#     image_size=(img_height, img_width),
#     batch_size=batch_size,
#     crop_to_aspect_ratio=True)


    
train_ds = tf.keras.utils.image_dataset_from_directory(
    "Modified_Training_Data/Combined_Training_sets",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    crop_to_aspect_ratio=True)

    


val_ds = tf.keras.utils.image_dataset_from_directory(
    "Modified_Training_Data/Combined_Training_sets",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    crop_to_aspect_ratio=True)




augment_images = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.4),
    layers.RandomZoom(0.4),
    layers.RandomBrightness(factor=0.4), ## POTENTIALLY LOWER TO PREVENT BIAS TO OVERLY BRIGHT IMAGES??
    layers.RandomContrast(factor=0.4), ## THESE TERMS WERE 2
])


# random_brightness_intensity = keras.Sequential([
#     layers.RandomBrightness(factor=0.1),
#     layers.RandomSaturation(factor=0.1),
# ])





model = models.Sequential()

model.add(augment_images)



# Data Scaling Layer
# model.add(generate_new_crops)

# # Image augmentation layer
# model.add(random_brightness_intensity)

# # normalization layer
model.add(layers.Rescaling(1./255))


# model.add(layers.Conv2D(16, (3, 3), input_shape=(img_height, img_width, 3)))
# model.add(layers.BatchNormalization())
# model.add(layers.Activation('relu'))
# model.add(layers.MaxPooling2D((2, 2)))


# model.add(layers.Dropout(0.2))



model.add(layers.Conv2D(32, (3, 3),  input_shape=(img_height, img_width, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))


model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(32, (3, 3),  input_shape=(img_height, img_width, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))


model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))


# model.add(layers.Dropout(0.1))


model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
# model.add(layers.MaxPooling2D((2, 2)))


# model.add(layers.Dropout(0.1))


model.add(layers.Conv2D(128, (3, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
# model.add(layers.MaxPooling2D((2, 2)))


# model.add(layers.Conv2D(64, (3, 3), activation='relu'))


model.add(layers.GlobalMaxPooling2D())

model.add(layers.Dropout(0.5))


model.add(layers.Dense(1, activation='sigmoid'))

# model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(), ## play around with this
              metrics=['accuracy'])

model.summary()


loss_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='weights/weights.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

testing_callback = TestingScript()

# number of epochs?
history = model.fit(train_ds, epochs=30, 
                    validation_data=val_ds,
                    callbacks=[loss_callback,
                               testing_callback])


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.show()

test_loss, test_acc = model.evaluate(val_ds, verbose=2)

print(f"test loss {test_loss}")
print(f"test acc {test_acc}")