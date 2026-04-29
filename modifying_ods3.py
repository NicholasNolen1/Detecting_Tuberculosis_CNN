import pandas as pd
import os
import shutil
from pathlib import Path
from glob import glob
import random
import cv2


dir_root = 'Training_and_Test_Data/Bacterium_Quantity_Dataset/ODS3'
annotation_root = os.path.join(dir_root, 'Annotated_ODS3', 'ANNOTATIONS')
output_dir = 'Modified_Training_Data/ODS3'
crop_size = 244

num_negative = 0
num_positive = 0

negative_to_use = 4

img_files = glob(os.path.join(dir_root, "S*/*.bmp"))
print(f"found {len(img_files)} files")

for img in img_files:
    
    paths = img.split(os.sep)
    img_folder = paths[-2]
    img_filename = paths[-1]
    
    csv_filename = img_filename.replace('.bmp','.csv')

    csv_path = os.path.join(annotation_root, img_folder, "ImageAnnotation", csv_filename)
    
    
    if not os.path.exists(img):
        print(f"Skipping {img_filename}, image not found at {img}")
        continue
        
    image = cv2.imread(img)
    
    
    if image is None:
        continue
    height, width, _ = image.shape
    
    if os.path.exists(csv_path) :
    
        df = pd.read_csv(csv_path, sep=r',', header=None, names = ['type', 'x', 'y'])
        if df.empty:
            continue
        
            
        for i, row in df.iterrows():
            center_x = int(row['x'])
            center_y = int(row['y'])
            label = str(row['type']).lower()
            
            
            if 'b' in label or 'bc' in label:
                num_positive = num_positive + 1
                label = 'Positive'
            else:
                num_negative = num_negative + 1
                label = 'Negative'
                
            y1 = max(0, center_y - (crop_size // 2))
            y2 = y1 + crop_size
            x1 = max(0, center_x - (crop_size // 2))
            x2 = x1 + crop_size
            
            
            # Adjust for out of bounds
            if y2 > height:
                y2 = height
                y1 = max(0, height - crop_size)
            if x2 > width:
                x2 = width
                x1 = max(0, width - crop_size)
                
            crop = image[int(y1):int(y2), int(x1):int(x2)]
            
            save_folder = os.path.join(output_dir, label)
            os.makedirs(save_folder, exist_ok = True)
            
            base_name = os.path.splitext(csv_filename)[0]
            save_path = os.path.join(save_folder, f"{base_name}_crop{i}.jpg")
            cv2.imwrite(save_path, crop)

            print(f"saved file {base_name}_crop{i}.jpg to dir {label}")
            
    else :
        for i in range(negative_to_use):
            x = random.randint(crop_size//2, width - crop_size//2)
            y = random.randint(crop_size//2, height - crop_size//2)

            crop = image[y - (crop_size//2): y + (crop_size//2)
                         , x - (crop_size//2): x + (crop_size//2)]
            
            num_negative = num_negative + 1
     
            save_folder = os.path.join(output_dir, 'Negative')
            os.makedirs(save_folder, exist_ok = True)
            
            base_name = os.path.splitext(csv_filename)[0]
            save_path = os.path.join(save_folder, f"{base_name}_crop{i}.jpg")
            cv2.imwrite(save_path, crop)
            
            print(f"saved negative file {base_name}_crop{i}.jpg to dir Negative")



print(num_negative, num_positive)