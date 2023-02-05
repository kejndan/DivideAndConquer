import torch
import os
from torchvision import transforms
from PIL import Image

class OnlineProductsDataset(torch.utils.data.Dataset):
    def __init__(self, ds_path, mode, transforms_mode='test', train_st_class=0, train_fin_class=11318,
                 test_st_class=11318, test_fin_class=22634,
                 nb_train_imgs=59551, nb_test_imgs=60502):
        self.mode = mode
        self.transforms_mode = transforms_mode
        self.sz_crop = 224
        self.sz_resize = 256
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if self.mode == 'train':
            self.classes = range(train_st_class, train_fin_class)
        else:
            self.classes = range(test_st_class, test_fin_class)
        self.ds_path = ds_path
        self.labels, self.img_paths, self.idxs = list(), list(), list()

        im_paths_list = os.path.join(ds_path, f'Ebay_{self.mode}.txt')
        with open(im_paths_list) as f:
            f.readline()
            idx, nb_imgs = 0, 0

            for (image_id, class_id, _, path) in map(str.split, f):
                nb_imgs += 1
                if int(class_id) - 1 in self.classes:
                    self.img_paths.append(os.path.join(self.ds_path, path))
                    self.labels.append(int(class_id) - 1)
                    self.idxs.append(idx)
                    idx += 1

        if self.mode == 'train':
            assert nb_imgs == nb_train_imgs
        else:
            assert nb_imgs == nb_test_imgs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        im = Image.open(self.img_paths[index])
        if len(list(im.split())) == 1:
            im = im.convert('RGB')
        im = self.apply_augmentation(im)
        return im, self.labels[index], index

    def apply_augmentation(self, im):
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
        return transforms_(im)

    def get_nb_classes(self):
        return len(self.classes)

    def get_label(self, index):
        return self.labels[index]

    def get_within_indexes(self, indexes):
        self.labels = [self.labels[i] for i in indexes]
        self.idxs = [self.idxs[i] for i in indexes]
        self.img_paths = [self.img_paths[i] for i in indexes]
