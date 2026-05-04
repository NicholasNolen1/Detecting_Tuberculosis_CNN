import cv2
import os
import numpy as np
from glob import glob


def crop_image(input_path, output_dir, crop_size):
    os.makedirs(output_dir, exist_ok = True)

    image = cv2.imread(input_path)
    if image is None:
        print(f"Can't find image at : {input_path}")
        return

    h, w, _ = image.shape
    
    
    name, _ = os.path.splitext(os.path.basename(input_path))
    
    output_folder = os.path.join(output_dir, name)
    # print(output_folder)
    
    count = 0
    

    # make a directory to contain the crops which represent the image
    os.makedirs(output_folder, exist_ok = True)

    for y in range(0, h - crop_size + 1, crop_size):
        for x in range(0, w - crop_size + 1, crop_size):
            crop = image[y:y+crop_size, x:x+crop_size]
            
            file_name = f"crop_{y}_{x}.png"
            cv2.imwrite(os.path.join(output_folder, file_name), crop)
            
            count = count + 1
        


input_folder = 'Training_and_Test_Data/Positive_Negative_Dataset'
output_folder = 'Modified_Training_Data/Positive_Negative_Dataset'
crop_size = 224


positive_images = glob(os.path.join(input_folder, "Positive", "*.jpg"))
positive_output = os.path.join(output_folder, "Positive")

negative_images = glob(os.path.join(input_folder, "Negative", "*.jpg"))
negative_output = os.path.join(output_folder, "Negative")


for img in positive_images:
    crop_image(img, positive_output, crop_size)

for img in negative_images:
    crop_image(img, negative_output, crop_size)


