import os
from PIL import Image
import numpy as np
import cv2
import math
import torchvision
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, rotate
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.camera_stats import RING_CAMERA_LIST
import random

from .utils import IMAGE_WIDTH, IMAGE_HEIGHT, ARGOVERSE_CLASS_NAMES
from ..utils import decode_binary_labels


class ArgoverseMapDataset(Dataset):
    def __init__(self, argo_loaders, label_root, image_size=[960, 600], log_names=None, is_train=True, labeled_data=True, label_percent=1.0):

        self.label_root = label_root
        self.image_size = image_size

        self.examples = []
        self.calibs = dict()

        # Preload training examples from Argoverse train and test sets
        self.loaders = argo_loaders
        
        self.is_train = is_train
        self.labeled_data = labeled_data
        self.label_percent = label_percent
        
        for split, loader in self.loaders.items():  # 'train', lodaer1; 'val': loader2
            self.preload(split, loader, log_names)  # load train or val dataset
        
        print('len of {} {} set = {}'.format('labeled' if labeled_data else 'unlabeled', 'training' if is_train else 'testing', len(self.examples)))
        

    def preload(self, split, loader, log_names=None):
        # Iterate over sequences
        for log in loader:
            # Check if the log is within the current dataset split
            logid = log.current_log

            if log_names is not None and logid not in log_names:
                continue

            self.calibs[logid] = dict()
            for camera, timestamps in log.image_timestamp_list_sync.items():

                if camera not in RING_CAMERA_LIST:
                    continue

                # Load image paths
                if self.is_train and (self.label_percent < 1):  # training set under semi-setting
                    n = round(len(timestamps) * self.label_percent)
                    if self.labeled_data:
                        for timestamp in timestamps[:n]:
                            self.examples.append((timestamp, split, logid, camera))
                    else:
                        for timestamp in timestamps[n:]:
                            self.examples.append((timestamp, split, logid, camera))
                else:
                    for timestamp in timestamps:
                        self.examples.append((timestamp, split, logid, camera))


    def __len__(self):
        return len(self.examples)


    def __getitem__(self, index):
        # Get the split, log and camera ids corresponding to the given timestamp
        timestamp, split, log, camera = self.examples[index]

        image1 = self.load_image(split, log, camera, timestamp)
        label1, mask1 = self.load_labels(split, log, camera, timestamp)
        calib = self.load_calib(split, log, camera)

        if self.is_train:  # train data
            image2, label2, mask2, theta = self.conjoint_rotation(image1, label1, mask1, calib)
            
            image1 = cv2.resize(image1, tuple(self.image_size))  # h x w x 3
            image1 = torch.from_numpy(image1.transpose(2, 0, 1).astype(np.float32)) / 255.0  # 3 x h x w
            
            image2 = cv2.resize(image2, tuple(self.image_size))  # h x w x 3
            image2 = torch.from_numpy(image2.transpose(2, 0, 1).astype(np.float32)) / 255.0  # 3 x h x w
            
            calib[0] *= self.image_size[0] / IMAGE_WIDTH
            calib[1] *= self.image_size[1] / IMAGE_HEIGHT
            calib = torch.from_numpy(calib.astype(np.float32))
            
            if self.labeled_data:  # labeled train data
                if random.random() > 0.5:
                    return image1, label1, mask1, calib  # original data
                else:
                    if index < 100:
                        print('Rotated labeled data! Theta =', float(theta))
                    return image2, label2, mask2, calib  # rotated data
            else:  # unlabeled train data
                if random.random() > 0.5:
                    return image1, calib  # original data
                else:
                    if index < 100:
                        print('Rotated unlabeled data! Theta =', float(theta))
                    return image2, calib  # original data
            
        image1 = cv2.resize(image1, tuple(self.image_size))  # h x w x 3
        image1 = torch.from_numpy(image1.transpose(2, 0, 1).astype(np.float32)) / 255.0  # 3 x h x w
        calib[0] *= self.image_size[0] / IMAGE_WIDTH
        calib[1] *= self.image_size[1] / IMAGE_HEIGHT
        calib = torch.from_numpy(calib.astype(np.float32))
        
        return image1, label1, mask1, calib  # val


    def load_image(self, split, log, camera, timestamp):
        # Load image
        loader = self.loaders[split]
        image = loader.get_image_at_timestamp(timestamp, camera, log)  # rgb, 1200 x 1920
        return image
    

    def load_calib(self, split, log, camera):
        # Get the loader for the current split
        loader = self.loaders[split]
        # Get intrinsics matrix and rescale to account for downsampling
        calib = loader.get_calibration(camera, log).K[:, :3].astype(np.float32)
        return calib
    

    def load_labels(self, split, log, camera, timestamp):
        # Construct label path from example data
        label_path = os.path.join(self.label_root, split, log, camera, f'{camera}_{timestamp}.png')
        
        # Load encoded label image as a torch tensor
        encoded_labels = to_tensor(Image.open(label_path)).long()

        # Decode to binary labels
        num_class = len(ARGOVERSE_CLASS_NAMES)  # 8
        labels = decode_binary_labels(encoded_labels, num_class+1)
        
        # 'labels' include: drivable_area, vehicle, pedestrian, large_vehicle, bicycle, bus, trailer, motorcycle
        labels, mask = labels[:-1].float(), (~labels[-1]).float()  # 0/1
        
        return labels, mask


    def conjoint_rotation(self, image, bev_label, mask, K):  # h x w x 3, 5 x 196 x 200, 196 x 200, 3 x 3
        fx, cx = K[0, 0], K[0, 2]
        fy, cy = K[1, 1], K[1, 2]
        theta = np.random.uniform(-35, +35)  # default: -20 degrees ~ +20 degrees

        cosTheta = math.cos(theta * math.pi / 180)
        sinTheta = math.sin(theta * math.pi / 180)

        h, w = image.shape[0], image.shape[1]
    
        # source pixels
        pts_src = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        x_z = (pts_src[:, 0] - cx) / fx
        y_z = (pts_src[:, 1] - cy) / fy

        # destination pixels
        pts_dst = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        pts_dst[:, 0] = fx * (x_z * cosTheta - sinTheta) / (cosTheta + x_z * sinTheta) + cx
        pts_dst[:, 1] = fy * y_z / (cosTheta + x_z * sinTheta) + cy

        # compute Homography matrix using matching relationship
        homography, status = cv2.findHomography(pts_src, pts_dst)

        # warp image using pure rotation
        image = cv2.warpPerspective(image, homography, (w, h), borderMode=cv2.BORDER_REPLICATE)

        bev_label = rotate(bev_label, angle=-theta, center=(99.5, -4))
        mask = rotate(mask.unsqueeze(0), angle=-theta, center=(99.5, -4))

        return image, bev_label, mask[0], torch.tensor(theta)  # h x w x 3, n x 196 x 200, 196 x 200

      
      