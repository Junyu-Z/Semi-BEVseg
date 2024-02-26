import random
import torch
from torch.utils.data import Dataset


class AugmentedLabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, label, mask, calib = self.dataset[index]

        # Apply data augmentation
        if random.random() > 0.5:
            if index < 100:
                print('Flipped labeled data!')
            image = torch.flip(image, (-1,))
            label = torch.flip(label, (-1,))
            mask = torch.flip(mask, (-1,))
            width = image.shape[-1]
            calib[0, 2] = width - calib[0, 2]

        return image, label, mask, calib
         

class AugmentedUnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image1, label1, mask1, image2, label2, mask2, calib, theta = self.dataset[index]

        # Apply data augmentation
        if random.random() > 0.5:
            if index < 100:
                print('Flipped unlabeled data!')
            image1 = torch.flip(image1, (-1,))
            label1 = torch.flip(label1, (-1,))
            mask1 = torch.flip(mask1, (-1,))
            image2 = torch.flip(image2, (-1,))
            label2 = torch.flip(label2, (-1,))
            mask2 = torch.flip(mask2, (-1,))
            
            width = image1.shape[-1]
            calib[0, 2] = width - calib[0, 2]
            theta = -theta

        return image1, label1, mask1, image2, label2, mask2, calib, theta



