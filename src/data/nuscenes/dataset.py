import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import cv2
import math
import torchvision
import PIL
from nuscenes import NuScenes
from torchvision.transforms.functional import to_tensor, rotate
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import random

from .utils import CAMERA_NAMES, NUSCENES_CLASS_NAMES, iterate_samples, make_transform_matrix
from ..utils import decode_binary_labels


class NuScenesMapDataset(Dataset):
    def __init__(self, nuscenes, map_root, image_size=(800, 600), scene_names=None, is_train=True, labeled_data=True, label_percent=1.0, enable_conjoint_rotataion=False):

        self.nuscenes = nuscenes
        self.map_root = os.path.expandvars(map_root)
        self.image_size = image_size

        # Preload the list of tokens in the dataset
        self.tokens = []  # len of training_tokens = 168048, len of testing_tokens = 35886
        self.gt_poses = []  # len = 168048
        
        self.is_train = is_train
        self.labeled_data = labeled_data
        self.label_percent = label_percent
        self.enable_conjoint_rotataion = enable_conjoint_rotataion
        
        # Allow PIL to load partially corrupted images
        # (otherwise training crashes at the most inconvenient possible times!)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        self.get_tokens(scene_names)
        print('len of {} {} set = {}'.format('labeled' if labeled_data else 'unlabeled', 'training' if is_train else 'testing', len(self.tokens)))


    def get_tokens(self, scene_names=None):
        # Iterate over scenes
        labeled_sample_per_scene = int(self.label_percent * 40)
        for scene in self.nuscenes.scene:
            # Ignore scenes which don't belong to the current split
            if scene_names is not None and scene['name'] not in scene_names:
                continue
            num_of_sample = 0
            for sample in iterate_samples(self.nuscenes, scene['first_sample_token']):
                if self.is_train and self.label_percent < 1:  # training set under semi-setting
                    num_of_sample += 1
                    if (self.labeled_data is True) and (num_of_sample == labeled_sample_per_scene + 1):
                        break
                    elif (self.labeled_data is False) and (num_of_sample <= labeled_sample_per_scene):
                        continue
                # Iterate over cameras
                for camera in CAMERA_NAMES:
                    self.tokens.append(sample['data'][camera])

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        token = self.tokens[index]

        ### load image ###
        image1 = cv2.imread(self.nuscenes.get_sample_data_path(token), -1)  # BGR, 900 x 1600 x 3
        image1 = image1[..., ::-1]  # BGR --> RGB
        
        ### load label ###
        label1, mask1 = self.load_labels(token)  # n x 196 x 200, 196 x 200
        ### load calib ###
        sample_data = self.nuscenes.get('sample_data', token)
        sensor = self.nuscenes.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        calib = sensor['camera_intrinsic']  # 3 x 3
        calib = np.array(calib)
        
        if self.is_train:  # train data
            image2, label2, mask2, theta = self.conjoint_rotation(image1, label1, mask1, calib)
            
            image1 = cv2.resize(image1, tuple(self.image_size))  # h x w x 3
            image1 = torch.from_numpy(image1.transpose(2, 0, 1).astype(np.float32)) / 255.0  # 3 x h x w
            
            image2 = cv2.resize(image2, tuple(self.image_size))  # h x w x 3
            image2 = torch.from_numpy(image2.transpose(2, 0, 1).astype(np.float32)) / 255.0  # 3 x h x w
            
            calib[0] *= self.image_size[0] / sample_data['width']
            calib[1] *= self.image_size[1] / sample_data['height']
            calib = torch.from_numpy(calib.astype(np.float32))  # 3 x 3
            
            if self.labeled_data:  # labeled train data
                if self.enable_conjoint_rotataion and random.random() > 0.5:
                    if index < 100: # for debugging
                        print('Rotated labeled data! Theta =', float(theta))
                    return image2, label2, mask2, calib  # rotated data
                else:
                    return image1, label1, mask1, calib  # original data
            else:  # unlabeled train data
                if self.enable_conjoint_rotataion and random.random() > 0.5:
                    if index < 100: # for debugging
                        print('Rotated unlabeled data! Theta =', float(theta))
                    return image2, calib  # rotated data
                else:
                    return image1, calib  # original data

        image1 = cv2.resize(image1, tuple(self.image_size))  # h x w x 3
        image1 = torch.from_numpy(image1.transpose(2, 0, 1).astype(np.float32)) / 255.0  # 3 x h x w
        
        calib[0] *= self.image_size[0] / sample_data['width']
        calib[1] *= self.image_size[1] / sample_data['height']
        calib = torch.from_numpy(calib.astype(np.float32))  # 3 x 3
        
        return image1, label1, mask1, calib  # test data
    

    def load_labels(self, token):
        # Load label image as a torch tensor
        label_path = os.path.join(self.map_root, token + '.png')
        encoded_labels = to_tensor(Image.open(label_path)).long()  # 1 x 196 x 200, 15bit

        # Decode to binary labels
        num_class = len(NUSCENES_CLASS_NAMES)  # 14
        labels = decode_binary_labels(encoded_labels, num_class + 1)  # 15 x 196 x 200, type=bool, denotes for existence of 15 categories
        
        # 'labels' include: drivable_area, ped_crossing, walkway, carpark, car, truck, bus, trailer, construction_vehicle, pedestrian, motorcycle, bicycle, traffic_cone, barrier
        labels, mask = labels[:-1].float(), (~labels[-1]).float()  # 0/1

        return labels, mask


    def conjoint_rotation(self, image, bev_label, mask, K):  # h x w x 3, n x 196 x 200, 196 x 200, 3 x 3
        fx, cx = K[0, 0], K[0, 2]
        fy, cy = K[1, 1], K[1, 2]
        theta = np.random.uniform(-35, +35)  # default: -35 degrees ~ +35 degrees

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
        image = cv2.warpPerspective(image, homography, (w, h), borderMode=cv2.BORDER_REPLICATE)  # BORDER_REPLICATE, BORDER_REFLECT

        bev_label = rotate(bev_label, angle=-theta, center=(99.5, -4))
        mask = rotate(mask.unsqueeze(0), angle=-theta, center=(99.5, -4))

        return image, bev_label, mask[0], torch.tensor(theta)  # h x w x 3, n x 196 x 200, 196 x 200

      
    def cut_out(self, image, bev_label, mask, K):  # h x w x 3, n x 196 x 200, 196 x 200, 3 x 3
        h, w = image.shape[0], image.shape[1]
        length = 100
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = int(np.clip(y - length/2, 0, h))
        y2 = int(np.clip(y + length/2, 0, h))
        x1 = int(np.clip(x - length/2, 0, w))
        x2 = int(np.clip(x + length/2, 0, w))
        image[y1: y2, x1: x2] = 0
        
        return image, bev_label, mask, torch.tensor(0.0)  # h x w x 3, n x 196 x 200, 196 x 200
      
      
    def random_erasing(self, image, bev_label, mask, K):  # h x w x 3, n x 196 x 200, 196 x 200, 3 x 3
        h, w = image.shape[0], image.shape[1]
        s = (0.02, 0.4)
        r = (0.3, 1/0.3)
        Se = random.uniform(*s) * h * w
        re = random.uniform(*r)
        He = int(round(math.sqrt(Se * re)))
        We = int(round(math.sqrt(Se / re)))
        xe = random.randint(0, w)
        ye = random.randint(0, h)
        if xe + We <= w and ye + He <= h:
            image[ye: ye + He, xe: xe + We] = np.random.randint(low=0, high=255, size=(He, We, 3))
        
        return image, bev_label, mask, torch.tensor(0.0)  # h x w x 3, n x 196 x 200, 196 x 200

      
      