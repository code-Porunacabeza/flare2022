from dataset.transform import crop, hflip, normalize, resize, blur, cutout
import nrrd
import math
import os
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):
    def __init__(self, root, mode, num_class, labeled_id_path=None, unlabeled_id_path=None, pseudo_mask_path=None,transforms=None):
        """

        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        :param labeled_id_path: path of labeled image ids, needed in train or semi_train mode.
        :param unlabeled_id_path: path of unlabeled image ids, needed in semi_train or label mode.
        :param pseudo_mask_path: path of generated pseudo masks, needed in semi_train mode.
        """

        self.root = root
        self.mode = mode

        self.num_classes=num_class
        self.pseudo_mask_path = pseudo_mask_path
        self.transforms=transforms

        if mode == 'semi_train':
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
            self.ids = \
                self.labeled_ids * math.ceil(len(self.unlabeled_ids) / len(self.labeled_ids)) + self.unlabeled_ids

        else:
            if mode == 'val':
                id_path = 'dataset/splits/val.txt'
            elif mode == 'label':
                id_path = unlabeled_id_path
            elif mode == 'train':
                id_path = labeled_id_path

            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
    def make_mask(self,mask):
        mask_list = []
        for i in range(self.num_classes):
            m = np.zeros_like(mask)
            if i==0:
                mask_list.append(m)
                continue
            m[np.where(mask == i)] = 1
            mask_list.append(m)
        arr=np.dstack(mask_list)
        return arr

    def normalization_imgs(self,imgs):
        ''' centering and reducing data structures '''
        imgs = imgs.astype(np.float32, copy=False)
        mean = np.mean(imgs)  # mean for data centering
        std = np.std(imgs)  # std for data normalization
        if np.int32(std) != 0:
            imgs -= mean
            imgs /= std
        return imgs
    def __getitem__(self, item):
        id = self.ids[item]
        img = nrrd.read(os.path.join(self.root, id.split(' ')[0]))[0]
        img = img[:, :, np.newaxis]
        img=self.normalization_imgs(img)

        if self.mode == 'val' :
            mask = nrrd.read(os.path.join(self.root, id.split(' ')[1]))[0]
            if self.transforms is not None:
                augmented = self.transforms(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            #img = img.astype('float32')
            img = img.transpose(2, 0, 1)
            mask = self.make_mask(mask)
            mask = mask.astype('float32')
            mask = mask.transpose(2, 0, 1)

            return img, mask, id
        if self.mode=='label':
            #img = img.astype('float32')
            img = img.transpose(2, 0, 1)
            return img, img, id
        if self.mode == 'train' or (self.mode == 'semi_train' and id in self.labeled_ids):
            mask = nrrd.read(os.path.join(self.root, id.split(' ')[1]))[0]
        else:
            # mode == 'semi_train' and the id corresponds to unlabeled image
            fname = os.path.basename(id)
            mask = nrrd.read(os.path.join(self.pseudo_mask_path, fname))[0]


        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        #img = img.astype('float32')
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32')
        mask = self.make_mask(mask)
        mask = mask.transpose(2, 0, 1)
        return img, mask

    def __len__(self):
        return len(self.ids)
