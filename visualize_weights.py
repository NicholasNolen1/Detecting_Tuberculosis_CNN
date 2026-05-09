
import tensorflow as tf

from keras.preprocessing import image
import numpy as np

from keras import models

import os

import math 

import matplotlib.pyplot as plt


# positive from ODS3
# image_path = 'Modified_Training_Data/ODS3/Positive/S18TR1EDFT042_crop26.jpg'

# negative from positive negative dataset
image_path = 'Modified_Training_Data/ODS3/Positive/S18TR1EDFT042_crop9.jpg'


# Pre-processing the image
img = image.load_img(image_path, target_size = (224, 224))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis = 0)
img_tensor = img_tensor / 255.

# Print image tensor shape
print(img_tensor.shape)

# Print image
plt.imshow(img_tensor[0])
plt.show()


weights_file = 'pretrained_weights/weights-100-epochs.keras'

model = tf.keras.models.load_model(weights_file)

# model.predict(img_tensor)


# Outputs of the 8 layers, which include conv2D and max pooling layers
layer_outputs = [layer.output for layer in model.layers[:22]]
activation_model = models.Model(inputs = model.inputs, outputs = layer_outputs)


activations = activation_model.predict(img_tensor)

for n, (layer, activation) in enumerate(zip(model.layers[:22], activations)):
    print(f"{n}: Layer: {layer.name} Shape {activation.shape}")
    
    num_channels = activation.shape[-1]
    cols = 8
    
    rows = math.ceil(num_channels / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), gridspec_kw={'wspace':0.01, 'hspace':0.01})
    
    fig.canvas.manager.set_window_title( f'Layer: {layer.name} | Shape {activation.shape}')

    
    axes = axes.flatten()
    
    for i in range(num_channels):
        ax = axes[i]
        ax.matshow(activation[0, :, :, i], cmap='viridis')
        ax.axis('off')
        
    for j in range(num_channels, len(axes)):
        axes[j].axis('off')
    

    os.makedirs('activation-images', exist_ok=True)

    fig.savefig(f"activation-images/activations_layer_{n}_{layer.name}.jpeg")
    plt.close(fig)
    # plt.tight_layout()
    # plt.show()



# Getting Activations of first layer
# first_layer_activation = activations[4]

# shape of first layer activation
# print(first_layer_activation.shape)





# # 6th channel of the image after first layer of convolution is applied
# plt.matshow(first_layer_activation[0, :, :, 3], cmap ='viridis')

# # 15th channel of the image after first layer of convolution is applied
# plt.matshow(first_layer_activation[0, :, :, 15], cmap ='viridis')

# plt.show()

layer_names = []
