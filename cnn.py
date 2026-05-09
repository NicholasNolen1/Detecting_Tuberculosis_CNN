import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from testing import TestingScript, PlotData

import pandas as pd
import numpy as np

import sys
import os



## EXPERIMENT WITH THESE:
batch_size = 64
img_height = 224
img_width = 224


##SPLIT DATA INTO TRAIN AND TEST
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



model.add(layers.Input(shape=(img_height, img_width, 3)))

model.add(augment_images)



# Data Scaling Layer
# model.add(generate_new_crops)

# # Image augmentation layer
# model.add(random_brightness_intensity)

# # normalization layer
model.add(layers.Rescaling(1./255))


# model.add(layers.Conv2D(16, (3, 3), ))
# model.add(layers.BatchNormalization())
# model.add(layers.Activation('relu'))
# model.add(layers.MaxPooling2D((2, 2)))


# model.add(layers.Dropout(0.2))


model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))


model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(32, (3, 3)))
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


os.makedirs('plots', exist_ok=True)
    

# plot every 25 epochs
figure_callback = PlotData(testing_callback, frequency=2)

# number of epochs?
history = model.fit(train_ds, epochs=100, # make 30 
                    validation_data=val_ds,
                    callbacks=[loss_callback,
                               testing_callback,
                               figure_callback])


fig1, ax_1 = plt.subplots()
ax_1.plot(history.history['accuracy'], label='accuracy')
ax_1.plot(history.history['val_accuracy'], label = 'val_accuracy')
ax_1.set_xlabel('Epoch')
ax_1.set_ylabel('Accuracy')
ax_1.set_ylim([0.5, 1])
ax_1.legend(loc='lower right')
fig1.savefig('plots/final_accuracy.png')



fig4, ax_4 = plt.subplots()
ax_4.plot(history.history['val_loss'], label='val_loss', color = 'blue')
ax_4.set_xlabel('Epoch')
ax_4.set_ylabel('Loss')
ax_4.set_title('Validation Loss Over Time')
ax_4.legend(loc='upper right')
fig4.savefig('plots/final_loss.png')


fig5, ax_5 = plt.subplots()
ax_5.plot(history.history['val_accuracy'], label = 'val_accuracy')
ax_5.set_xlabel('Epoch')
ax_5.set_ylabel('Accuracy')
ax_5.set_title('Validation Accuracy Over Time')
ax_5.legend(loc='upper right')
fig5.savefig('plots/final_val_accuracy.png')


plt.show()


test_loss, test_acc = model.evaluate(val_ds, verbose=2)

print(f"test loss {test_loss}")
print(f"test acc {test_acc}")