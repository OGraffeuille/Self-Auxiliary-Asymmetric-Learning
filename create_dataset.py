import os
import cv2
import random
import torch
import fnmatch

import pandas as pd
import numpy as np
import panoptic_parts as pp
import torch.utils.data as data
import matplotlib.pylab as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
from torchvision.transforms import InterpolationMode

from PIL import Image
from torchvision.datasets import CIFAR100


# CelebA dependencies
import imageio
import PIL
import torch
import re
import glob

from torch.utils import data


class DataTransform(object):
    def __init__(self, scales, crop_size, is_disparity=False):
        self.scales = scales
        self.crop_size = crop_size
        self.is_disparity = is_disparity

    def __call__(self, data_dict):
        if type(self.scales) == tuple:
            # Continuous range of scales
            sc = np.random.uniform(*self.scales)

        elif type(self.scales) == list:
            # Fixed range of scales
            sc = random.sample(self.scales, 1)[0]

        raw_h, raw_w = data_dict['im'].shape[-2:]
        resized_size = [int(raw_h * sc), int(raw_w * sc)]
        i, j, h, w = 0, 0, 0, 0  # initialise cropping coordinates
        flip_prop = random.random()

        for task in data_dict:
            if len(data_dict[task].shape) == 2:   # make sure single-channel labels are in the same size [H, W, 1]
                data_dict[task] = data_dict[task].unsqueeze(0)

            # Resize based on randomly sampled scale
            if task in ['im', 'noise']:
                data_dict[task] = transforms_f.resize(data_dict[task], resized_size, InterpolationMode.BILINEAR)
            elif task in ['normal', 'depth', 'seg', 'part_seg', 'disp']:
                data_dict[task] = transforms_f.resize(data_dict[task], resized_size, InterpolationMode.NEAREST)

            # Add padding if crop size is smaller than the resized size
            if self.crop_size[0] > resized_size[0] or self.crop_size[1] > resized_size[1]:
                right_pad, bottom_pad = max(self.crop_size[1] - resized_size[1], 0), max(self.crop_size[0] - resized_size[0], 0)
                if task in ['im']:
                    data_dict[task] = transforms_f.pad(data_dict[task], padding=(0, 0, right_pad, bottom_pad),
                                                       padding_mode='reflect')
                elif task in ['seg', 'part_seg', 'disp']:
                    data_dict[task] = transforms_f.pad(data_dict[task], padding=(0, 0, right_pad, bottom_pad),
                                                       fill=-1, padding_mode='constant')  # -1 will be ignored in loss
                elif task in ['normal', 'depth', 'noise']:
                    data_dict[task] = transforms_f.pad(data_dict[task], padding=(0, 0, right_pad, bottom_pad),
                                                       fill=0, padding_mode='constant')  # 0 will be ignored in loss

            # Random Cropping
            if i + j + h + w == 0:  # only run once
                i, j, h, w = transforms.RandomCrop.get_params(data_dict[task], output_size=self.crop_size)
            data_dict[task] = transforms_f.crop(data_dict[task], i, j, h, w)

            # Random Flip
            if flip_prop > 0.5:
                data_dict[task] = torch.flip(data_dict[task], dims=[2])
                if task == 'normal':
                    data_dict[task][0, :, :] = - data_dict[task][0, :, :]

            # Final Check:
            if task == 'depth':
                data_dict[task] = data_dict[task] / sc

            if task == 'disp':  # disparity is inverse depth
                data_dict[task] = data_dict[task] * sc

            if task in ['seg', 'part_seg']:
                data_dict[task] = data_dict[task].squeeze(0)
        return data_dict


class NYUv2(data.Dataset):
    """
    NYUv2 dataset, 3 tasks + 1 generated useless task
    Included tasks:
        1. Semantic Segmentation,
        2. Depth prediction,
        3. Surface Normal prediction,
        4. Noise prediction [to test auxiliary learning, purely conflict gradients]
    """
    def __init__(self, root, partition="train", augmentation=False, noise=False):
        self.partition = partition
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation
        self.noise = noise

        # read the data file
        if self.partition in ["train", "vali"]:
            self.data_path = root + '/train'
        elif self.partition == "test":
            self.data_path = root + '/test'
        else:
            assert False, "Invalid partition name: {}".format(partition)

        # calculate data length
        data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))
        if partition == "train":
            self.data_len = int(data_len * 0.8)
        elif partition == "vali":
            self.data_len = int(data_len * 0.2)
            self.data_skip = int(data_len * 0.8)
        elif partition == "test":
            self.data_len = data_len
        else:
            assert False, "Invalid partition name: {}".format(partition)
        
        if noise:
            self.noise = torch.rand(self.data_len, 1, 288, 384)

    def __getitem__(self, index):
        index_orig = index
        if self.partition == "vali":
            index += self.data_skip
            
        # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0)).float()
        semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index))).long()
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0)).float()
        normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0)).float()
                
        if self.noise:
            noise = self.noise[index_orig].float()
            data_dict = {'im': image, 'seg': semantic, 'depth': depth, 'normal': normal, 'noise': noise}
        else:
            data_dict = {'im': image, 'seg': semantic, 'depth': depth, 'normal': normal}

        # apply data augmentation if required
        if self.augmentation:
            data_dict = DataTransform(crop_size=[288, 384], scales=[1.0, 1.2, 1.5])(data_dict)

        im = 2. * data_dict.pop('im') - 1.  # normalised to [-1, 1]
        return im, data_dict

    def __len__(self):
        return self.data_len


class CityScapes(data.Dataset):
    """
    CityScapes dataset, 3 tasks + 1 generated useless task
    Included tasks:
        1. Semantic Segmentation,
        2. Part Segmentation,
        3. Disparity Estimation (Inverse Depth),
        4. Noise prediction [to test auxiliary learning, purely conflict gradients]

    Note: 
    Original implementations use test set as the validation set. 
    We instead use 20% of training data as validation.
    """
    def __init__(self, root, partition="train", augmentation=False, noise=False):
        self.partition = partition
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation
        self.noise = noise

        # read the data file
        if self.partition in ["train", "vali"]:
            self.data_path = root + '/train'
        elif self.partition == "test":
            self.data_path = root + '/test'
        else:
            assert False, "Invalid partition name: {}".format(partition)

        # calculate data length
        data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.png'))
        if partition == "train":
            self.data_len = int(data_len * 0.8)
        elif partition == "vali":
            self.data_len = int(data_len * 0.2)
            self.data_skip = int(data_len * 0.8)
        elif partition == "test":
            self.data_len = data_len
        else:
            assert False, "Invalid partition name: {}".format(partition)
        
        if noise:
            self.noise = torch.rand(self.data_len, 1, 256, 256) if self.partition in ["train", "vali"] else torch.rand(self.data_len, 1, 256, 512)

    def __getitem__(self, index):
        index_orig = index
        if self.partition == "vali":
            index += self.data_skip
        # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(plt.imread(self.data_path + '/image/{:d}.png'.format(index)), -1, 0)).float()
        disparity = cv2.imread(self.data_path + '/depth/{:d}.png'.format(index), cv2.IMREAD_UNCHANGED).astype(np.float32)
        disparity = torch.from_numpy(self.map_disparity(disparity)).unsqueeze(0).float()
        #depth = self.disp2depth(disparity)
        seg = np.array(Image.open(self.data_path + '/seg/{:d}.png'.format(index)), dtype=float)
        seg = torch.from_numpy(self.map_seg_label(seg)).long()
        part_seg = np.array(Image.open(self.data_path + '/part_seg/{:d}.tif'.format(index)))
        part_seg = torch.from_numpy(self.map_part_seg_label(part_seg)).long()

        if self.noise:
            noise = self.noise[index_orig].float()
            data_dict = {'im': image, 'seg': seg, 'part_seg': part_seg, 'disp': disparity, 'noise': noise}
        else:
            data_dict = {'im': image, 'seg': seg, 'part_seg': part_seg, 'disp': disparity}

        # apply data augmentation if required
        if self.augmentation:
            pass
            #data_dict = DataTransform(crop_size=[256, 256], scales=[1.0])(data_dict)

        im = 2. * data_dict.pop('im') - 1.  # normalised to [-1, 1]
        return im, data_dict

    def map_seg_label(self, mask):
        # source: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        mask_map = np.zeros_like(mask)
        mask_map[np.isin(mask, [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30])] = -1
        mask_map[np.isin(mask, [7])] = 0
        mask_map[np.isin(mask, [8])] = 1
        mask_map[np.isin(mask, [11])] = 2
        mask_map[np.isin(mask, [12])] = 3
        mask_map[np.isin(mask, [13])] = 4
        mask_map[np.isin(mask, [17])] = 5
        mask_map[np.isin(mask, [19])] = 6
        mask_map[np.isin(mask, [20])] = 7
        mask_map[np.isin(mask, [21])] = 8
        mask_map[np.isin(mask, [22])] = 9
        mask_map[np.isin(mask, [23])] = 10
        mask_map[np.isin(mask, [24])] = 11
        mask_map[np.isin(mask, [25])] = 12
        mask_map[np.isin(mask, [26])] = 13
        mask_map[np.isin(mask, [27])] = 14
        mask_map[np.isin(mask, [28])] = 15
        mask_map[np.isin(mask, [31])] = 16
        mask_map[np.isin(mask, [32])] = 17
        mask_map[np.isin(mask, [33])] = 18
        return mask_map

    def map_part_seg_label(self, mask):
        # https://panoptic-parts.readthedocs.io/en/stable/api_and_code.html
        # https://arxiv.org/abs/2004.07944
        mask = pp.decode_uids(mask, return_sids_pids=True)[-1]
        mask_map = np.zeros_like(mask)  # background
        mask_map[np.isin(mask, [2401, 2501])] = 1    # human/rider torso
        mask_map[np.isin(mask, [2402, 2502])] = 2    # human/rider head
        mask_map[np.isin(mask, [2403, 2503])] = 3    # human/rider arms
        mask_map[np.isin(mask, [2404, 2504])] = 4    # human/rider legs
        mask_map[np.isin(mask, [2601, 2701, 2801])] = 5  # car/truck/bus windows
        mask_map[np.isin(mask, [2602, 2702, 2802])] = 6  # car/truck/bus wheels
        mask_map[np.isin(mask, [2603, 2703, 2803])] = 7  # car/truck/bus lights
        mask_map[np.isin(mask, [2604, 2704, 2804])] = 8  # car/truck/bus license_plate
        mask_map[np.isin(mask, [2605, 2705, 2805])] = 9  # car/truck/bus chassis
        return mask_map

    def map_disparity(self, disparity):
        # https://github.com/mcordts/cityscapesScripts/issues/55#issuecomment-411486510
        # remap invalid points to -1 (not to conflict with 0, infinite depth, such as sky)
        disparity[disparity == 0] = -1
        # reduce by a factor of 4 based on the rescaled resolution
        disparity[disparity > -1] = (disparity[disparity > -1] - 1) / (256 * 4)

        return disparity
    
    def disp2depth(self, disparity):
        return (0.209313 * 2262.52) / disparity

    def __len__(self):
        return self.data_len


class CIFAR100MTL(CIFAR100):
    """
    CIFAR100 dataset, 20 tasks (grouped by coarse labels)
    Each task is a 5-label classification, with 2500 training and 500 testing number of data for each task.
    Modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(self, root, subset_id=0, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100MTL, self).__init__(root, train, transform, target_transform, download)
        # define coarse label maps
        coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13])

        self.coarse_targets = coarse_labels[self.targets]

        # filter the data and targets for the desired subset
        self.data = self.data[self.coarse_targets == subset_id]
        self.targets = np.array(self.targets)[self.coarse_targets == subset_id]

        # remap fine labels into 5-class classification
        self.targets = np.unique(self.targets, return_inverse=True)[1]

        # update semantic classes
        self.class_dict = {
            "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
            "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
            "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
            "food containers": ["bottle", "bowl", "can", "cup", "plate"],
            "fruit and vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
            "household electrical device": ["clock", "computer_keyboard", "lamp", "telephone", "television"],
            "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
            "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
            "large carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
            "large man-made outdoor things": ["bridge", "castle", "house", "road", "skyscraper"],
            "large natural outdoor scenes": ["cloud", "forest", "mountain", "plain", "sea"],
            "large omnivores and herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
            "medium-sized mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
            "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
            "people": ["baby", "boy", "girl", "man", "woman"],
            "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
            "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
            "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
            "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
            "vehicles 2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
        }

        self.subset_class = list(self.class_dict.keys())[subset_id]
        self.classes = self.class_dict[self.subset_class]





# Code adapted from https://github.com/isl-org/MultiObjectiveOptimization/blob/master/multi_task/
# Oli: Made changes: .png -> .jpg
class CelebA(data.Dataset):
    def __init__(self, root, split="train", is_transform=True, img_size=None, augmentations=None, data_frac=1.0):
        """__init__
        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """

        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.data_frac = data_frac
        self.n_classes = 40
        self.img_size = img_size
        if type(img_size) == int:
            self.img_size = (img_size, img_size)
        self.mean = np.array([73.15835921, 82.90891754, 72.39239876])  # TODO(compute this mean)
        self.files = {}
        self.labels = {}

        self.label_file = self.root + "/Anno/list_attr_celeba.txt"
        label_map = {}
        with open(self.label_file, 'r') as l_file:
            labels = l_file.read().split('\n')[2:-1]
        for label_line in labels:
            f_name = re.sub('jpg', 'jpg', label_line.split(' ')[0])
            # f_name = re.sub('jpg', 'PNG', label_line.split(' ')[0])
            label_txt = list(map(lambda x: int(x), re.sub('-1', '0', label_line).split()[1:]))
            label_map[f_name] = label_txt

        self.all_files = glob.glob(self.root + '/Img/img_align_celeba/*.jpg')
        # self.all_files = glob.glob(self.root+'/Img/img_align_celeba_png/*.png')
        with open(root + '//Eval/list_eval_partition.txt', 'r') as f:
            fl = f.read().split('\n')
            fl.pop()
            if 'train' in self.split:
                selected_files = list(filter(lambda x: x.split(' ')[1] == '0', fl))
            elif 'val' in self.split:
                selected_files = list(filter(lambda x: x.split(' ')[1] == '1', fl))
            elif 'test' in self.split:
                selected_files = list(filter(lambda x: x.split(' ')[1] == '2', fl))
            selected_file_names = list(map(lambda x: re.sub('jpg', 'jpg', x.split(' ')[0]), selected_files))
            # selected_file_names = list(map(lambda x:re.sub('jpg', 'png', x.split(' ')[0]), selected_files))

        if os.name == 'nt': # Windows
            base_path = self.all_files[0].split('\\')[0]
            self.files[self.split] = list(map(lambda x: '/'.join([base_path, x]), set(map(lambda x:x.split('\\')[-1], self.all_files)).intersection(set(selected_file_names))))
            self.labels[self.split] = list(map(lambda x: label_map[x], set(map(lambda x:x.split('\\')[-1], self.all_files)).intersection(set(selected_file_names))))
        else:
            base_path = '/'.join(self.all_files[0].split('/')[:-1])
            self.files[self.split] = list(map(lambda x: '/'.join([base_path, x]), set(map(lambda x: x.split('/')[-1], self.all_files)).intersection(set(selected_file_names))))
            self.labels[self.split] = list(map(lambda x: label_map[x], set(map(lambda x: x.split('/')[-1], self.all_files)).intersection(set(selected_file_names))))

        self.task_names = {f'class_{i:d}': 2 for i in range(40)}
        self.class_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
                            'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                            'Bushy_Eyebrows',
                            'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
                            'High_Cheekbones',
                            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
                            'Pale_Skin',
                            'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
                            'Wavy_Hair',
                            'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                            'Wearing_Necktie', 'Young']

        if len(self.files[self.split]) < 2:
            raise Exception("No files for split=[%s] found in %s" % (self.split, self.root))

    def __len__(self):
        """__len__"""
        if self.data_frac == 1.0 or self.split == "test":
            n = len(self.files[self.split])
        else:
            n = int(len(self.files[self.split]) * self.data_frac)
        return n

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        label = self.labels[self.split][index]
        label = {t: label[i] for i, t in enumerate(self.task_names)}
        img = np.asarray(imageio.imread(img_path))

        if self.augmentations is not None:
            img = self.augmentations(np.array(img, dtype=np.uint8))

        if self.is_transform:
            img = self.transform_img(img)

        #return [img] + label
        return img, label

    def transform_img(self, img):
        """transform
        Mean substraction, remap to [0,1], channel order transpose to make Torch happy
        """
        img = img[:, :, ::-1]
        if self.img_size is not None:
            img = np.array(PIL.Image.fromarray(img).resize((self.img_size[0], self.img_size[1]), resample=PIL.Image.NEAREST))
        img = img.astype(np.float64)
        img -= self.mean
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img


class RobotArm(data.Dataset):

    def __init__(self, root, tasks, n_train_per_task, seed=0, split='train'):

        # To fix: currently generates all 10 tasks, then returns mostly empty labels for each task
        # need to instead discard generated data not from corresponding tasks (but still generate  N_TASKS=10 tasks data)
        # pass train_tasks as argument and filter that, check gloria code

        self.root = root
        self.task_ids = list(set([int(t.split(".aux")[0].split("_")[-1]) for t in tasks.keys()]))
        self.n_tasks = len(self.task_ids)
        self.n_train_per_task = n_train_per_task
        self.seed = seed
        self.split = split

        # Don't change. Variables to generate data. Won't necessarily use all the data.
        self.N_TASKS = 100
        self.N_TRAIN_PER_TASK = 1000 # train and vali data
        self.N_TEST_PER_TASK = 1000


        self.n_train = int(self.n_tasks * self.n_train_per_task * 0.8)
        self.n_vali = int(self.n_tasks * self.n_train_per_task * 0.2)
        self.n_test = self.n_tasks * self.N_TEST_PER_TASK

        assert self.n_train_per_task <= 1000, "ERROR: We only generate 1000 training examples per task, cannot use more."
        assert self.n_tasks <= 100, "ERROR: We only generate data for 100 tasks, cannot use more."

        self.data_folderpath = os.path.join(self.root, f"robotarm_2D_3DF_{self.seed}")

        if not os.path.exists(self.data_folderpath):
            self._generate_data(self.seed)
        else:
            self._load_data()

        # Generated data has 100 tasks, we may use less
        inds_tasks = np.isin(self.T, self.task_ids).flatten()

        # Select dataset inds
        # Note that data is generated T = [0, 1, ... 99, 0, 1, ...]
        if self.split == 'train':
            slice_split = slice(0, self.n_train)
        elif self.split == 'vali':
            slice_split = slice(self.n_train, self.n_train+self.n_vali)
        elif self.split == 'test':
            # Use end of dataset to have consistent test set across data params
            slice_split = slice(self.N_TRAIN_PER_TASK * self.n_tasks)

        self.X = self.X[inds_tasks][slice_split]
        self.y = self.y[inds_tasks][slice_split]
        self.T = self.T[inds_tasks][slice_split]
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).float()
        self.T = torch.from_numpy(self.T).int()

    def __getitem__(self, index):

        X = self.X[index]
        y = self.y[index]

        label = {f"regression_{i}": y if i == self.T.flatten()[index]
                 else torch.full(y.shape, float('nan')) for i in self.task_ids}

        return X, label

    def __len__(self):
        if self.split == 'train':
            return self.n_train
        elif self.split == 'vali':
            return self.n_vali
        elif self.split == 'test':
            return self.n_test

    def _generate_data(self, seed):

        np.random.seed(seed)

        n_tasks = self.N_TASKS
        n_test_per_task = self.N_TRAIN_PER_TASK
        n_train_per_task = self.N_TEST_PER_TASK

        dof = 3
        joint_length = [0, 1]


        n_data_per_task = n_train_per_task + n_test_per_task
        n_train = n_train_per_task * n_tasks
        n_test = n_test_per_task * n_tasks

        shp = (n_train + n_test,)  # How many data points to generate

        lengths = np.random.uniform(low=joint_length[0], high=joint_length[1], size=(n_tasks, dof))

        angles = np.stack([np.random.uniform(low=-np.pi, high=0, size=shp),
                           np.random.uniform(low=-np.pi, high=0, size=shp),
                           np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=shp)], axis=1)
        x = np.zeros(shp)
        y = np.zeros(shp)
        self.T = np.tile(np.arange(n_tasks), n_data_per_task).reshape(-1, 1)
        for i in range(dof):
            x += np.tile(lengths[:, i], n_data_per_task) * np.cos(np.sum(angles[:, :i + 1], axis=1))
            y += np.tile(lengths[:, i], n_data_per_task) * np.sin(np.sum(angles[:, :i + 1], axis=1))

        X = np.stack([x, y], axis=1)
        y = angles

        X_mu = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        self.X = (X - X_mu) / X_std

        y_mu = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        self.y = (y - y_mu) / y_std

        os.mkdir(self.data_folderpath)
        np.save(os.path.join(self.data_folderpath, "X.npy"), self.X)
        np.save(os.path.join(self.data_folderpath, "y.npy"), self.y)
        np.save(os.path.join(self.data_folderpath, "T.npy"), self.T)

    def _load_data(self):
        self.X = np.load(os.path.join(self.data_folderpath, "X.npy"))
        self.y = np.load(os.path.join(self.data_folderpath, "y.npy"))
        self.T = np.load(os.path.join(self.data_folderpath, "T.npy"))


class Gloria(data.Dataset):

    def __init__(self, root, label, tasks, seed=0, split='train'):
        self.root = root
        self.task_ids = list(set([int(t.split(".aux")[0].split("_")[-1]) for t in tasks.keys()]))
        self.n_tasks = len(self.task_ids)
        self.seed = seed
        self.split = split
        self.label = label

        # Set seed
        np.random.seed(seed)

        # Don't change: Total number of tasks available in the dataset
        # (affects dataset generation because we make tasks balanced in size)
        self.N_TASKS = 10

        self.data_folderpath = os.path.join(self.root, f"data_{label}.csv")
        assert os.path.exists(self.data_folderpath), f"ERROR: data not found: {self.data_folderpath}"
        assert self.n_tasks <= self.N_TASKS, "ERROR: Too many tasks used."

        # Filter dataset for the tasks we use
        self.data = pd.read_csv(self.data_folderpath)
        self.data = self.data[self.data["task"] < self.N_TASKS]
        smallest_task_size = self.data.groupby('task').size().min()
        self.data = self.data.groupby('task').head(smallest_task_size).reset_index(drop=True)

        # Filter for only tasks which are actually included in the data
        self.data = self.data[self.data['task'].isin(self.task_ids)].reset_index(drop=True)

        # Re-order to have consistently distributed tasks
        new_order = [ind_task * smallest_task_size + ind_data for ind_data in range(smallest_task_size) for ind_task in range(self.n_tasks)]
        self.data = self.data.iloc[new_order].reset_index(drop=True)

        self.n_train = int(len(self.data.index) * 0.6)
        self.n_vali = int(len(self.data.index) * 0.2)
        self.n_test = int(len(self.data.index) * 0.2)

        inds = np.arange(len(self.data.index))
        if split == 'train':
            inds = inds[:self.n_train]
        elif split == 'vali':
            inds = inds[self.n_train:self.n_train+self.n_vali]
        elif split == 'test':
            inds = inds[self.n_train+self.n_vali:]
        else:
            raise ValueError(f"ERROR: Unknown split {split}.")

        self.X = self.data.loc[inds, [str(i) for i in range(16)]].values
        self.y = self.data.loc[inds, 'label'].values
        self.T = self.data.loc[inds, 'task'].values
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).float().view(-1, 1)
        self.T = torch.from_numpy(self.T).int()


    def __getitem__(self, index):

        X = self.X[index]
        y = self.y[index]

        # Return nans if task has no label (not all X values have a corresponding labels for all tasks)
        label = {f"regression_{i}": y if i == self.T.flatten()[index]
        else torch.full(y.shape, float('nan')) for i in self.task_ids}

        return X, label

    def __len__(self):
        return self.X.shape[0]
