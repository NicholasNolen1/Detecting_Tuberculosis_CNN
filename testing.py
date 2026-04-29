import cv2
import os
import numpy as np
import tensorflow as tf
from glob import glob
from tensorflow.keras.models import Sequential


class TestingScript(tf.keras.callbacks.Callback):
    def __init__(self ):
        super().__init__()
        self.crop_size = 244
        self.threshold = 0.5 # for classification
        self.n_samples = 100
        self.input_folder = 'Training_and_Test_Data/Positive_Negative_Dataset'


    def get_crops(self, image_path):

        image = cv2.imread(image_path)
        if image is None:
            print(f"Can't find image at : {image_path}")
            return []

        h, w, _ = image.shape
        
                    
        cropped_images = []

        for y in range(0, h - self.crop_size + 1, self.crop_size):
            for x in range(0, w - self.crop_size + 1, self.crop_size):
                crop = image[y:y+self.crop_size, x:x+self.crop_size]

                crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                
                cropped_images.append(crop)
                
                
        return cropped_images




    def test_slide(self, image_path):     
        crops = self.get_crops(image_path)
        
        if not crops:
            return 0, 0, 0 # no crops extracted
        
        batch = np.array(crops).astype('float32')
                
        prediction = self.model.predict(batch, len(batch), verbose = 0)
        
        total = 0
        # counting the number of values predicted to be above the threshold likely to contain tb
        pos_count = np.sum(prediction.flatten() >= self.threshold)
        
        slide_prediction = 0
        
        
        # if more than half of the slides are predicted to be positive its classified as positive
        if pos_count >= len(crops) / 2.0:
            slide_prediction = 1
        
        
        return slide_prediction, pos_count, len(crops)      
            
    def on_epoch_end(self,epoch, logs=None):
        positive_images = glob(os.path.join(self.input_folder, "Positive", "*.jpg"))

        negative_images = glob(os.path.join(self.input_folder, "Negative", "*.jpg"))

        # # weights_file = 'weights/32Start-256End.keras'
        # weights_file = 'weights/weights-simple-model.keras'

        # model = tf.keras.models.load_model(weights_file)


        num_pos_samples = min(self.n_samples, len(positive_images)) # default to n_samples unless positive_images is too small
        samples = np.random.choice(positive_images, num_pos_samples, replace=False)

        # print(f"testing: {weights_file}")

        positive_sum_slide = 0.0
        total_positive_crops = 0
        total_num_crops = 0
        
        # random permutation of 100 images
        
        for positive_image in samples:
                                    
            # returns an average of the predictions
            slide_prediction, pos_count, num_crops_tested = self.test_slide(positive_image)
            
            positive_sum_slide += slide_prediction
            total_positive_crops += pos_count
            total_num_crops += num_crops_tested

        print(f"\ntesting sub croppings of slides labeled positive:")
        print(f"total amount of crops predicted positive: {total_positive_crops}")
        print(f"percent slides predicted positive: {positive_sum_slide / num_pos_samples:.2f}")

        num_neg_samples = min(self.n_samples, len(negative_images)) # default to n_samples unless positive_images is too small
        neg_samples = np.random.choice(negative_images, num_neg_samples, replace=False)

        # print(f"testing: {weights_file}")

        negative_sum_slide = 0.0
        total_positive_crops_negative_run = 0
        total_num_crops_negative_run = 0
        
        # random permutation of 100 images
        
        for negative_image in neg_samples:
                                    
            # returns an average of the predictions
            slide_prediction, neg_count, num_crops_tested = self.test_slide(negative_image)
            
            negative_sum_slide += slide_prediction
            total_positive_crops_negative_run += neg_count
            total_num_crops_negative_run += num_crops_tested

            
        print("\ntesting sub croppings of slides labeled negative:")
        # subtracting total_positive_crops_negative_run from total crops to get total negative crops
        print(f"total number of crops predicted negative: {(total_num_crops_negative_run - total_positive_crops_negative_run )}")
        print(f"percent of slides predicted negative: {( num_neg_samples - negative_sum_slide ) / num_neg_samples:.2f}")
        
        print(f"\ntotal accurate slide prediction percentage: {( num_neg_samples - negative_sum_slide + positive_sum_slide) / (num_neg_samples + num_pos_samples):.2f}")
