pip install albumentations opencv-python matplotlib
import os
import cv2
import random
import numpy as np
from albumentations import (
    Compose, RandomRotate90, Flip, Transpose,
    ElasticTransform, GridDistortion, OpticalDistortion,
    RandomBrightnessContrast, RandomGamma, ShiftScaleRotate
)
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# 1) Paths
INPUT_DIR  = r"C:\Users\15613\Downloads\chest_xray\train"
OUTPUT_DIR = r"C:\Users\15613\Downloads\chest_xray_augmented\train"

# 2) Choose how many augmented samples per original image
AUG_PER_IMAGE = 5

# 3) Define advanced medical-style augmentations
aug_pipeline = Compose([
    ElasticTransform(alpha=1.0, sigma=50, alpha_affine=50, p=0.5),
    GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    OpticalDistortion(distort_limit=0.3, shift_limit=0.1, p=0.5),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    Flip(p=0.5),
    Transpose(p=0.5),
])

# 4) Utility to make output subfolders
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 5) Process each class folder
for label in os.listdir(INPUT_DIR):
    class_in  = os.path.join(INPUT_DIR, label)
    class_out = os.path.join(OUTPUT_DIR, label)
    ensure_dir(class_out)
    
    for fname in os.listdir(class_in):
        if not fname.lower().endswith(('.png','.jpg','.jpeg')):
            continue
        
        img_path = os.path.join(class_in, fname)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        
     
        cv2.imwrite(os.path.join(class_out, fname), image)
        
  
        for i in range(AUG_PER_IMAGE):
            augmented = aug_pipeline(image=image)['image']
            new_name = f"{os.path.splitext(fname)[0]}_aug{i+1}.png"
            cv2.imwrite(os.path.join(class_out, new_name), augmented)

print(OUTPUT_DIR)
sample_folder = OUTPUT_DIR
sample_imgs = []
for root, _, files in os.walk(sample_folder):
    for f in files:
        if 'aug' in f:
            sample_imgs.append(os.path.join(root, f))
            if len(sample_imgs) >= 6: break
    if len(sample_imgs) >= 6: break

plt.figure(figsize=(12, 6))
for i, img_path in enumerate(sample_imgs):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.subplot(2, 3, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(os.path.basename(img_path))
    plt.axis('off')
plt.tight_layout()
plt.show()
