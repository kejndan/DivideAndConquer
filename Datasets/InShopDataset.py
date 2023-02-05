import torch
import os
from torchvision import transforms
from PIL import Image


class InShopDataset(torch.utils.data.Dataset):
    def __init__(self, ds_path, transforms_mode='test', ds_info_path='Eval/list_eval_partition.txt',
                 mode='train', nb_train_images=25882,
                 nb_query_images=14218, nb_gallery_images=12612, nb_train_classes=3997, nb_query_classes=3985,
                 nb_gallery_classes=3985):

        ''' nb_all_images = 52712 '''
        self.ds_info_path = os.path.join(ds_path, ds_info_path)
        self.mode = mode
        self.transforms_mode = transforms_mode
        self.paths, self.labels, self.ids = [], [], []

        self.sz_crop = 224
        self.sz_resize = 256
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        with open(self.ds_info_path, 'r') as f:
            ds_info = f.readlines()[2:]

        for i, inf_ in enumerate(ds_info):
            inf_ = inf_.replace('\n', '').split()
            path, lbl, mode = inf_
            for el in [path, lbl, mode]:
                assert el != ''
            if mode == self.mode:
                self.paths.append(os.path.join(ds_path, path))
                self.labels.append(int(lbl[3:]))
                self.ids.append(i)

        ordered_labels = {lbl: i for i, lbl in enumerate(sorted(set(self.labels)))}
        self.labels = list(map(lambda l: ordered_labels[l], self.labels))
        if self.mode is 'train':
            nb_images, nb_classes = nb_train_images, nb_train_classes
        elif self.mode is 'query':
            nb_images, nb_classes = nb_query_images, nb_query_classes
        else:
            nb_images, nb_classes = nb_gallery_images, nb_gallery_classes
        assert len(self.labels) == nb_images
        assert len(set(self.labels)) == nb_classes

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
        self.paths[idx] = self.paths[idx].replace('/img/', '/Img/')
        assert os.path.exists(self.paths[idx])
        try:
            # assert os.path.exists(self.paths[idx])
            img = Image.open(self.paths[idx])
        except:
            print(self.paths[idx])

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
