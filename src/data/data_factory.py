import os
from torch.utils.data import DataLoader, RandomSampler
from .augmentation import *

from nuscenes import NuScenes
from .nuscenes.dataset import NuScenesMapDataset
from .nuscenes.splits import TRAIN_SCENES, VAL_SCENES, CALIBRATION_SCENES
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from .argoverse.dataset import ArgoverseMapDataset
from .argoverse.splits import TRAIN_LOGS, VAL_LOGS


#==================================================================================================
def build_fullNu_datasets(config):
    print('==> Loading NuScenes training dataset...')
    nuscenes = NuScenes(config.nuscenes_version, os.path.expandvars(config.nuscenes_dataroot))
    nuscenes_train_data = NuScenesMapDataset(nuscenes, config.nuscenes_label_root, config.img_size, TRAIN_SCENES, is_train=True, labeled_data=True)
    
    print('==> Loading NuScenes testing dataset...')
    nuscenes_test_data = NuScenesMapDataset(nuscenes, config.nuscenes_label_root, config.img_size, VAL_SCENES, is_train=False)
    
    return nuscenes_train_data, nuscenes_test_data


def build_semiNu_datasets(config):
    print('==> Loading NuScenes training dataset...')
    nuscenes = NuScenes(config.nuscenes_version, os.path.expandvars(config.nuscenes_dataroot))
    nuscenes_labeled_train_data = NuScenesMapDataset(nuscenes, config.nuscenes_label_root, config.img_size, TRAIN_SCENES, 
                                                     is_train=True, labeled_data=True, label_percent=config.label_percent)
    nuscenes_unlabeled_train_data = NuScenesMapDataset(nuscenes, config.nuscenes_label_root, config.img_size, TRAIN_SCENES, 
                                                       is_train=True, labeled_data=False, label_percent=config.label_percent)
    
    print('==> Loading NuScenes testing dataset...')
    nuscenes_test_data = NuScenesMapDataset(nuscenes, config.nuscenes_label_root, config.img_size, VAL_SCENES, is_train=False)
    
    return nuscenes_labeled_train_data, nuscenes_unlabeled_train_data, nuscenes_test_data
    

# def build_NuAndUnlabeledAr_datasets(config):
#     print('==> Loading NuScenes training dataset...')
#     nuscenes = NuScenes(config.nuscenes_version, os.path.expandvars(config.nuscenes_dataroot))
#     nuscenes_labeled_train_data = NuScenesMapDataset(nuscenes, config.nuscenes_label_root, config.img_size, TRAIN_SCENES, is_train=True, labeled_data=True)
    
#     print('==> Loading Argoverse training dataset...')
#     dataroot = os.path.expandvars(config.argoverse_dataroot)
#     # Load native argoverse splits
#     loaders = {
#         'train': ArgoverseTrackingLoader(os.path.join(dataroot, 'train')),
#         'val': ArgoverseTrackingLoader(os.path.join(dataroot, 'val'))
#     }
#     argoverse_unlabeled_train_data = ArgoverseMapDataset(loaders, config.argoverse_label_root, config.img_size, TRAIN_LOGS, is_train=True, labeled_data=False, label_percent=0)
    
#     print('==> Loading NuScenes testing dataset...')
#     nuscenes_test_data = NuScenesMapDataset(nuscenes, config.nuscenes_label_root, config.img_size, VAL_SCENES, is_train=False)
    
#     return nuscenes_labeled_train_data, argoverse_unlabeled_train_data, nuscenes_test_data
  

#==================================================================================================
'''
def build_fullAr_datasets(config):
    print('==> Loading Argoverse training dataset...')
    dataroot = os.path.expandvars(config.argoverse_dataroot)
    # Load native argoverse splits
    loaders = {
        'train': ArgoverseTrackingLoader(os.path.join(dataroot, 'train')),
        'val': ArgoverseTrackingLoader(os.path.join(dataroot, 'val'))
    }
    argoverse_train_data = ArgoverseMapDataset(loaders, config.argoverse_label_root, config.img_size, TRAIN_LOGS, is_train=True, labeled_data=True)
    
    print('==> Loading Argoverse testing dataset...')
    argoverse_test_data = ArgoverseMapDataset(loaders, config.argoverse_label_root, config.img_size, VAL_LOGS, is_train=False)

    return argoverse_train_data, argoverse_test_data


def build_semiAr_datasets(config):
    print('==> Loading Argoverse training dataset...')
    dataroot = os.path.expandvars(config.argoverse_dataroot)
    # Load native argoverse splits
    loaders = {
        'train': ArgoverseTrackingLoader(os.path.join(dataroot, 'train')),
        'val': ArgoverseTrackingLoader(os.path.join(dataroot, 'val'))
    }
    
    argoverse_labeled_train_data = ArgoverseMapDataset(loaders, config.argoverse_label_root, config.img_size, TRAIN_LOGS,
                                                       is_train=True, labeled_data=True, label_percent=config.label_percent)
    argoverse_unlabeled_train_data = ArgoverseMapDataset(loaders, config.argoverse_label_root, config.img_size, TRAIN_LOGS,
                                                         is_train=True, labeled_data=False, label_percent=config.label_percent)
    
    print('==> Loading Argoverse testing dataset...')
    argoverse_test_data = ArgoverseMapDataset(loaders, config.argoverse_label_root, config.img_size, VAL_LOGS, is_train=False)
    
    return argoverse_labeled_train_data, argoverse_unlabeled_train_data, argoverse_test_data
  
  
def build_ArAndUnlabeledNu_datasets(config):
    print('==> Loading Argoverse training dataset...')
    dataroot = os.path.expandvars(config.argoverse_dataroot)
    # Load native argoverse splits
    loaders = {
        'train': ArgoverseTrackingLoader(os.path.join(dataroot, 'train')),
        'val': ArgoverseTrackingLoader(os.path.join(dataroot, 'val'))
    }
    
    argoverse_labeled_train_data = ArgoverseMapDataset(loaders, config.argoverse_label_root, config.img_size, TRAIN_LOGS, is_train=True, labeled_data=True)
    
    print('==> Loading NuScenes training dataset...')
    nuscenes = NuScenes(config.nuscenes_version, os.path.expandvars(config.nuscenes_dataroot))
    nuscenes_unlabeled_train_data = NuScenesMapDataset(nuscenes, config.nuscenes_label_root, config.img_size, TRAIN_SCENES, is_train=True, labeled_data=False, label_percent=0)
    
    print('==> Loading Argoverse testing dataset...')
    argoverse_test_data = ArgoverseMapDataset(loaders, config.argoverse_label_root, config.img_size, VAL_LOGS, is_train=False)
    
    return argoverse_labeled_train_data, nuscenes_unlabeled_train_data, argoverse_test_data
'''
 