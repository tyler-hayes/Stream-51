import os
import os.path
import random

import pickle
from PIL import Image
import numpy as np
import torch.utils.data as data
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as transforms

def instance_ordering(data_list, seed):
    # organize data by video
    total_videos = 0
    new_data_list = []
    temp_video = []
    for x in data_list:
        if x[3] == 0:
            new_data_list.append(temp_video)
            total_videos += 1
            temp_video = [x]
        else:
            temp_video.append(x)
    new_data_list.append(temp_video)
    new_data_list = new_data_list[1:]
    # shuffle videos
    random.seed(seed)
    random.shuffle(new_data_list)
    # reorganize by clip
    data_list = []
    for v in new_data_list:
        for x in v:
            data_list.append(x)
    return data_list # , new_data_list


def class_ordering(data_list, class_type, seed):
    # organize by class
    new_data_list = []
    class_vids = []
    class_vids_len = []
    hist_data = {}
    for class_id in range(data_list[-1][0] + 1):
        class_data_list = [x for x in data_list if x[0] == class_id]
        if class_type == 'class_iid':
            # shuffle all class data
            random.seed(seed)
            random.shuffle(class_data_list)
        else:
            # shuffle clips within class
            class_data_list = instance_ordering(class_data_list, seed)
        new_data_list.append(class_data_list)
    # shuffle classes
    random.seed(seed)
    random.shuffle(new_data_list)
    # reorganize by class
    data_list = []
    for v in new_data_list:
        for x in v:
            data_list.append(x)
    return data_list


def make_dataset(data_list, ordering='class_instance', seed=666):
    """
    data_list
    for train: [class_id, clip_num, video_num, frame_num, bbox, file_loc]
    for test: [class_id, bbox, file_loc]
    """
    if not ordering or len(data_list[0]) == 3:  # cannot order the test set
        return data_list
    if ordering not in ['iid', 'class_iid', 'instance', 'class_instance']:
        raise ValueError('dataset ordering must be one of: "iid", "class_iid", "instance", or "class_instance"')
    if ordering == 'iid':
        # shuffle all data
        random.seed(seed)
        random.shuffle(data_list)
        return data_list
    elif ordering == 'instance':
        return instance_ordering(data_list, seed)
    elif 'class' in ordering:
        return class_ordering(data_list, ordering, seed)


class StreamDataset(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, pickle_file, ordering=None, transform=None, target_transform=None, bbox_crop=True,
                 ratio=1.10, seed=666):

        data_list = pickle.load(open(pickle_file, 'rb'))
        samples = make_dataset(data_list, ordering, seed=seed)

        self.root = root
        self.loader = default_loader

        self.samples = samples
        self.targets = [s[0] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

        self.bbox_crop = bbox_crop
        self.ratio = ratio

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        fpath, target = self.samples[index][-1], self.targets[index]
        sample = self.loader(os.path.join(self.root, fpath))
        if self.bbox_crop:
            bbox = self.samples[index][-2]
            cw = bbox[0] - bbox[1];
            ch = bbox[2] - bbox[3];
            center = [int(bbox[1] + cw / 2), int(bbox[3] + ch / 2)]
            bbox = [min([int(center[0] + (cw * self.ratio / 2)), sample.size[0]]),
                    max([int(center[0] - (cw * self.ratio / 2)), 0]),
                    min([int(center[1] + (ch * self.ratio / 2)), sample.size[1]]),
                    max([int(center[1] - (ch * self.ratio / 2)), 0])]
            sample = sample.crop((bbox[1],
                                  bbox[3],
                                  bbox[0],
                                  bbox[2]))

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)

def filter_by_class(labels, seen_classes):
    ixs = []
    for c in seen_classes:
        i = list(np.where(labels == c)[0])
        ixs += i
    return ixs

def get_stream60_data_loader(images_dir, pickle_prefix, training, ordering=None, batch_size=128, shuffle=False,
                             augment=False, num_workers=8, seen_classes=None, seed=200, ix=None):
    if training:
        split = 'train'
    else:
        split = 'test'

    # resize to 224, send to tensor, then normalize with standard imagenet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if training and augment:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

    dataset = StreamDataset(images_dir, images_dir + '/' + pickle_prefix + split + '_ssd.pkl',
                            ordering=ordering, transform=transform, bbox_crop=True, ratio=1.10, seed=seed)
    labels = np.array([t for t in dataset.targets])

    if seen_classes is not None:
        indices = filter_by_class(labels, seen_classes)
        sub = Subset(dataset, indices)
        loader = DataLoader(sub, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True, shuffle=shuffle)
    elif ix is not None:
        sub = Subset(dataset, ix)
        loader = DataLoader(sub, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True, shuffle=shuffle)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True, shuffle=shuffle)

    return loader
