
import tensorflow as tf
from glob import glob
from tensorflow.keras.models import Sequential
from testing import TestingScript

import numpy as np
import cv2
import os

from PIL import Image, ImageTk

from pathlib import Path
from itertools import cycle

import shutil

import tkinter as tk


class Application(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title("Slideshow")
        self.geometry("650x300")
        self.resizable(width=False, height=False)
        
        
        self.image_label_frame = tk.Frame(self)
        self.image_label_frame.pack(expand=True, fill="both", padx=10, pady=10)
        
        
        self.current_slide = tk.Label(self.image_label_frame)
        self.current_slide.pack(expand=True, fill="both", side=tk.LEFT, padx=10)
        
        self.description_text = tk.Label(self.image_label_frame, text="", wraplength=300)
        self.description_text.pack(side=tk.LEFT, padx=10)

        
        self.button_frame = tk.Frame(self)
        self.button_frame.pack(side=tk.BOTTOM, pady=5)
        
        
        self.prev_btn = tk.Button(self.button_frame, text="Prev",command=self.display_prev_slide)
        self.prev_btn.pack(side=tk.LEFT, padx=5)

        self.next_btn = tk.Button(self.button_frame, text="Next",command=self.display_next_slide)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        


    def set_image_directory(self, path, overall_data):
        image_paths = sorted(Path(path).glob("*.jpg"))
        
        self.images = []
        self.predictions = []
        self.data = overall_data
        
        for path in image_paths:
            
            self.predictions.append(float(path.name.replace(".jpg","").split("_")[-1]))
            
            photo = ImageTk.PhotoImage(Image.open(path))
            self.images.append((path.name,photo))
        
        self.index = 0

    def display_next_slide(self):
        if self.index < len(self.images):
            name, self.next_image = self.images[self.index]
            self.current_slide.config(image=self.next_image)
            
            self.prediction = self.predictions[self.index]            
            self.description_text.config(text=f"likelihood crop is pos: {self.prediction}\n\n{self.data}")

                        
            self.title(name)
            
            self.index += 1
        
    def display_prev_slide(self):
        if self.index  > 0:
            name, self.prev_image = self.images[self.index -1]
            self.current_slide.config(image=self.prev_image)

            self.prediction = self.predictions[self.index -1]            
            self.description_text.config(text=f"likelihood crop is pos: {self.prediction}\n\n{self.data}")
            
            self.title(name)
            
            self.index -= 1

    def start(self):
        self.display_next_slide()
        self.mainloop()



weights_file = 'pretrained_weights/weights-100-epochs.keras'

model = tf.keras.models.load_model(weights_file)


tester = TestingScript()


# run model on input image


def run_viewer(image, image_directory):
    crops = tester.get_crops(image)
            
    if not crops:
        print("An error occurred")

    batch = np.array(crops).astype('float32')
            
    prediction = model.predict(batch, len(batch), verbose = 0)

    total = 0
    # counting the number of values predicted to be above the threshold likely to contain tb
    threshold = .5
    pos_count = np.sum(prediction.flatten() >= threshold)

    sorted_crops = sorted(zip(prediction.flatten(), batch), reverse=True)


    positive_sum_slide = 0.0
    total_positive_crops = 0

    total_positive_crops = pos_count


    data = "----Threshold for predictions is .5----" + \
    f"\ntotal num crops predicted positive : {total_positive_crops}" +  \
    f"\npercent of crops predicted positive: {total_positive_crops / len(batch):.3f}" + \
    f"\n\ntotal num crops predicted negative : { len(batch) - total_positive_crops}" + \
    f"\npercent of crops predicted negative: { (len(batch) - total_positive_crops) / len(batch):.3f}" 



        


    os.makedirs(image_directory, exist_ok=True)
    print(f"Created new directory: {image_directory}")

    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Full path to output: {os.path.abspath(image_directory)}")

    for i, (prediction_val, crop) in enumerate(sorted_crops):
        filename = os.path.join(image_directory, f"crop_{str(i).zfill(3)}_pred_{prediction_val:.3f}.jpg")
        
        cv2.imwrite(filename, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        
        print(f"saved crop to: {filename}")
        

    slide_prediction = 0
    # if more than half of the slides are predicted to be positive its classified as positive
    if pos_count >= len(sorted_crops) / 2.0:
        slide_prediction = 1
        
    application = Application()
    application.set_image_directory(image_directory, data)
    application.start()
    application.mainloop()

    shutil.rmtree(image_directory)


# Positive Image:
positive_image = "Modified_Training_Data/ODS3/Positive/S18TR1EDFT042_crop9.jpg"

# Negative Image:
# negative_image = "Training_and_Test_Data/Positive_Negative_Dataset/Negative/240525_734_79.jpg"

negative_image = "Modified_Training_Data/TBDB/Negative/00208_5.jpg"
run_viewer(positive_image, 'output_crops/positive')

run_viewer(negative_image, 'output_crops/negative')

os.rmdir('output_crops')




# remove the organized_photos directory when done



