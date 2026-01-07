import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class VOCAugDataset(Dataset):
    def __init__(self, root, split='train_aug', transform=None):
        super(VOCAugDataset, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        
        self.images_dir = os.path.join(root, 'VOC2012', 'JPEGImages')
        self.split_txt_path = os.path.join(root, 'VOC2012', 'ImageSets', 'Segmentation', f'{split}.txt')

        
        if split == 'train_aug':
            self.masks_dir = os.path.join(root, 'VOC2012', 'SegmentationClassAug')
        else:
            self.masks_dir = os.path.join(root, 'VOC2012', 'SegmentationClass')
        # ======================

        if not os.path.exists(self.split_txt_path):
            raise FileNotFoundError(f"找不到列表文件: {self.split_txt_path}")
        if not os.path.exists(self.masks_dir):
             raise FileNotFoundError(f"找不到Mask文件夹: {self.masks_dir}")

        with open(self.split_txt_path, 'r') as f:
            self.file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]   
        print(f"[{split}] Loaded {len(self.file_names)} images from {self.split_txt_path}")
        self.classes = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        self.ignore_index = 255

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img_id = self.file_names[index]
        img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
        mask_path = os.path.join(self.masks_dir, f"{img_id}.png")
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path)) 
        
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]


        if not isinstance(img, torch.Tensor):

            img = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0
            
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.array(mask)).long()

        return img, mask