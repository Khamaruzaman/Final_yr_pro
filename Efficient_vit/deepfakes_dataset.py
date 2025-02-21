import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2 
import numpy as np

import uuid
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate

from transforms.albu import IsotropicResize

class DeepFakesDataset(Dataset):
    def __init__(self, images, labels, image_size, mode = 'train'):
        self.x = images
        self.y = torch.from_numpy(labels)
        self.image_size = image_size
        self.mode = mode
        self.n_samples = images.shape[0]
    
    def create_train_transforms(self, size):
        return Compose([
            ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
            GaussNoise(p=0.3),
            #GaussianBlur(blur_limit=3, p=0.05),
            HorizontalFlip(),
            OneOf([
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.4),
            ToGray(p=0.2),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ]
        )
        
    def create_val_transform(self, size):
        return Compose([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        ])

    def __getitem__(self, index):
        try:
            image = np.asarray(self.x[index])
            
            if self.mode == 'train':
                transform = self.create_train_transforms(self.image_size)
            else:
                transform = self.create_val_transform(self.image_size)
                    
            unique = uuid.uuid4()
            #cv2.imwrite("../dataset/augmented_frames/isotropic_augmentation/"+str(unique)+"_"+str(index)+"_original.png", image)
    
            image = transform(image=image)['image']
            
            #cv2.imwrite("../dataset/augmented_frames/isotropic_augmentation/"+str(unique)+"_"+str(index)+".png", image)
            
            return torch.tensor(image).float(), self.y[index]
        except Exception as e:
            print(f"Error loading image at index {index}: {e}")
            return torch.zeros((self.image_size, self.image_size, 3)), torch.tensor(0)  # Return dummy data


    def __len__(self):
        return self.n_samples

# import torch
# from torch.utils.data import Dataset
# import cv2
# import numpy as np
# from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, PadIfNeeded
# from transforms.albu import IsotropicResize

# class DeepFakesDataset(Dataset):
#     def __init__(self, image_paths, labels, image_size, mode='train'):
#         self.image_paths = image_paths  # Store file paths instead of image arrays
#         self.labels = torch.tensor(labels, dtype=torch.long)
#         self.image_size = image_size
#         self.mode = mode
#         self.n_samples = len(image_paths)  # Use length of paths, not images
    
#     def create_train_transforms(self, size):
#         return Compose([
#             HorizontalFlip(),
#             IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
#             PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
#             RandomBrightnessContrast(),
#         ])
    
#     def create_val_transform(self, size):
#         return Compose([
#             IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
#             PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
#         ])

#     def __getitem__(self, index):
#         image_path = self.image_paths[index]

#         if not isinstance(image_path, str):
#             print(f"Error: Expected string path, but got {type(image_path)} at index {index}")
#             return torch.zeros((self.image_size, self.image_size, 3)), self.labels[index]

#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Warning: Could not read image at path: {image_path}")
#             return torch.zeros((self.image_size, self.image_size, 3)), self.labels[index]

#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         transform = self.create_train_transforms(self.image_size) if self.mode == 'train' else self.create_val_transform(self.image_size)
#         image = transform(image=image)['image']

#         return torch.tensor(image).float(), self.labels[index]


#     def __len__(self):
#         return self.n_samples
