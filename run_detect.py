import photoshow

import tensorflow as tf
from glob import glob
from tensorflow.keras.models import Sequential
from testing import TestingScript

import numpy as np
import cv2
import os

from PIL import Image


weights_file = 'weights/weights.keras'

model = tf.keras.models.load_model(weights_file)


tester = TestingScript()


# run model on input image

# Positive Image:
positive_image = "/Users/nicholasnolen/Desktop/Intro to Machine Learning/Semester_Project/Training_and_Test_Data/Positive_Negative_Dataset/Positive/231226_736_5.jpg"


# Negative Image:
# /Users/nicholasnolen/Desktop/Intro to Machine Learning/Semester_Project/Training_and_Test_Data/Positive_Negative_Dataset/Negative/231228_416_41.jpg


positive_crops = tester.get_crops(positive_image)
        
if not positive_crops:
    print("An error occurred")

batch = np.array(positive_crops).astype('float32')
        
prediction = model.predict(batch, len(batch), verbose = 0)

total = 0
# counting the number of values predicted to be above the threshold likely to contain tb
threshold = .5
pos_count = np.sum(prediction.flatten() >= threshold)

sorted_crops = [x for _, x in sorted(zip(prediction.flatten(), batch), reverse=True)]

output_dir = "organized_photos"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created new directory: {output_dir}")

import os
print(f"Current Working Directory: {os.getcwd()}")
print(f"Full path to output: {os.path.abspath(output_dir)}")

for i, crop in enumerate(sorted_crops):
    filename = os.path.join(output_dir, f"crop_{str(i).zfill(3)}.jpg")
    
    cv2.imwrite(filename, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    
    print(f"saved crop to: {filename}")
    
    # image_to_save = image_path
    # image_to_save.save(os.path.join(output_dir, f"crop_{str(i).zfill(3)}.jpg"))


slide_prediction = 0
# if more than half of the slides are predicted to be positive its classified as positive
if pos_count >= len(sorted_crops) / 2.0:
    slide_prediction = 1
    



photoshow.present('./organized_photos')
