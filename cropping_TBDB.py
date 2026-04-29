import os
import cv2
import xml.etree.ElementTree as ET
from glob import glob

annotation_dir = 'Training_and_Test_Data/TBDB/Annotations_quarterly'
img_dir = 'Training_and_Test_Data/TBDB/JPEGS_quarterly'
output_dir = 'Modified_Training_Data/TBDB'
crop_size = 244

xml_files = glob(os.path.join(annotation_dir, "*.xml"))
print(f"Found {len(xml_files)} files!")

for xml in xml_files:
    tree = ET.parse(xml)
    root = tree.getroot()
    
    file_name = root.find('filename').text
    img_path = os.path.join(img_dir, file_name)
    
    if not os.path.exists(img_path):
        print(f"skipping {file_name}, not found")
        continue
    
    image = cv2.imread(img_path)
    height, width, _ = image.shape
    
    for i, obj in enumerate(root.findall('object')):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        
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
        
        class_path = os.path.join(output_dir, label)
        os.makedirs(class_path, exist_ok = True)
        
        base_name = os.path.splitext(file_name)[0]
        save_path = os.path.join(class_path, f"{base_name}_{i}.jpg")
        cv2.imwrite(save_path, crop)
        
        
# MANUALLY RENAME TBBacilly to Positive
# MANUALLY REMANE Debirs to Negative
# TODO: EVALUATE FOR OVERLAP, IF OVERLAP, LABEL POSITIVE