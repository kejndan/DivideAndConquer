import torch
import scipy
import os
import numpy as np
from torchvision import transforms
from PIL import Image


class CARSDataset(torch.utils.data.Dataset):
    def __init__(self, ds_path, mode, transforms_mode='test', nb_train_images=8054, nb_test_images=8131, nb_all=16185):

        self.ds_path = ds_path
        self.mode = mode
        self.transforms_mode = transforms_mode
        self.paths, self.labels, self.ids = [], [], []

        self.sz_crop = 224
        self.sz_resize = 256
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        mat = scipy.io.loadmat(os.path.join(ds_path, 'cars_annos.mat'))
        id = 0
        for annotation in mat['annotations'][0]:
            label = int(str(np.squeeze(annotation['class'])))
            if mode is 'train' and label < 99:
                self.paths.append(str(np.squeeze(annotation['relative_im_path'])))
                self.labels.append(int(str(np.squeeze(annotation['class']))))
                self.ids.append(id)
                id += 1
            elif mode is 'test' and label >= 99:
                self.paths.append(str(np.squeeze(annotation['relative_im_path'])))
                self.labels.append(int(str(np.squeeze(annotation['class']))))
                self.ids.append(id)
                id += 1

        if mode is 'train':
            assert len(self.labels) == nb_train_images
        else:
            assert len(self.labels) == nb_test_images

    def apply_augmentation(self, img):
        if self.transforms_mode == 'train':
            transforms_ = transforms.Compose([
                transforms.RandomResizedCrop(self.sz_crop),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean,
                    std=self.std,
                )
            ])
        else:
            transforms_ = transforms.Compose([
                transforms.Resize(self.sz_resize),
                transforms.CenterCrop(self.sz_crop),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean,
                    std=self.std,
                )
            ])
        return transforms_(img)

    def __len__(self):
        return len(self.labels)

    def get_nb_classes(self):
        return len(set(self.labels))

    def unique_labels(self):
        return set(self.labels)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.ds_path, self.paths[idx]))
        if len(list(img.split())) != 3:
            img = img.convert('RGB')
        img = self.apply_augmentation(img)
        return img, self.labels[idx], idx

    def get_label(self, idx):
        return self.labels[idx]

    def get_within_indexes(self, idxs):
        self.paths = [self.paths[i] for i in idxs]
        self.labels = [self.labels[i] for i in idxs]
        self.ids = [self.ids[i] for i in idxs]
