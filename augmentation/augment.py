import shutil
import os
import uuid
import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.RandomResizedCrop(size=[300, 300], scale=(0.8, 1.0)),
    A.Affine(
        scale=(0.8, 1.2),
        rotate=(-15, 15),
        p=0.7
    ),
    A.SquareSymmetry(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.1),
    A.HueSaturationValue(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.1, 0.15),
                        hole_width_range=(0.1, 0.15), p=0.3),
    ToTensorV2()
])

def load_and_transform(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    # Денормализуем и преобразуем в numpy
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_denorm = augmented['image'] * std + mean  # обратная нормализация
    img_np = img_denorm.permute(1, 2, 0).numpy()  # [H, W, C]
    img_np = (img_np * 255).astype(np.uint8)
    return img_np

if(__name__ == '__main__'):
    data_dir = "./data"
    augments_for_img = 1
    for subdir in os.listdir(data_dir):
        for file in os.listdir(os.path.join(data_dir, subdir)):
            filepath = os.path.join(data_dir, subdir, file)
            for i in range(augments_for_img):
                augment_filename = '.'.join(file.split(".")[:-1]) + "_augmented" + str(i) + "." + file.split(".")[-1]
                augment_filepath = os.path.join(data_dir, subdir, augment_filename)
                augmented_img = load_and_transform(filepath)
                cv2.imwrite(augment_filepath, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))