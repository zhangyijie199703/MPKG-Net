import os
import glob
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF

from torch.utils import data
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class TestFullSize(data.Dataset):

    def __init__(self, root, data_list, transform=None):
        super(TestFullSize, self).__init__()

        if isinstance(data_list, str):
            img_ids = [i_id.strip().split() for i_id in open(data_list)]
        elif isinstance(data_list,list):
            img_ids = []
            for idx, fp_list in enumerate(data_list):
                with open(fp_list) as f:
                    lines = f.readlines()
                    for line in lines:
                        img_ids.append(line.strip().split())
        else:
            raise TypeError(f'data_list type error: {type(data_list)}')

        self.files = []
        for item in img_ids:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            img_file = os.path.join(root, image_path)
            label_file = os.path.join(root, label_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
                })

        self.image_mean = [0.38083647, 0.33612353, 0.35943412]
        self.image_std = [0.10240941, 0.10278902, 0.10292039]

        # self.transform = transform
        # if self.transform is None:
        #     raise RuntimeError("Pytorch requires necessary transformtions!")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        data_item = self.files[index]
        print(data_item)
        image = Image.open(data_item['img']).convert('RGB')
        label = Image.open(data_item['label']).convert('RGB')
        # print(image.shape)
        # print(label.shape)

        image,  label = self.Normalize(image, label)
        image,  label = self.toTensor(image,  label)

        label = np.asarray(label, dtype=np.uint8)    
        label = self.encode_segmap(label).astype(np.uint8)  # numpy

        # array_image = np.asarray(image) # dtype: np.uint8
        # array_gt = np.asarray(gt) # dtype: np.uint8
        # array_gt = self.encode_segmap(array_gt)

        # image = self.transform(array_image.copy())
        # mask = torch.from_numpy(array_gt.copy()).long()

        image = torch.cat([image], dim=0)
        label = torch.from_numpy(label).type(torch.LongTensor)

        return image, label, data_item['name']

    def Normalize(self, image, label, div_std=False):
        image = np.array(image).astype(np.float32)

        image /= 255
        image -= self.image_mean

        if div_std == True:
            image /= self.std

        return image, label

    def toTensor(self, image, label):
        image = np.array(image).astype(np.float32).transpose((2, 0, 1))
        image = torch.from_numpy(image).type(torch.FloatTensor)

        return image, label

    def get_colormap(self):
        return np.array(
                        [
                            [255, 255, 255],
                            [0,   0,   255],
                            [0,   255, 255],
                            [0,   255, 0  ],
                            [255, 255, 0  ],
                            [255, 0,   0  ],
                            [0,   0,   0  ]
                        ]
                    )

    def encode_segmap(self, mask,):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
            (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_colormap()):
            if ii == 6:
                label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = 255
            else:
                label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
            # label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask


if __name__ == "__main__":

    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.3829, 0.3629, 0.3369], [0.1388, 0.1377, 0.1431]),
    ])

    dataset = TestFullSize(root="/media/ssd/syz_data/TGARSLetter/dataset/potsdam/", data_list='/media/ssd/syz_data/TGARSLetter/data/potsdam/potsdam_test_fullsize.txt',\
                        transform=T,)

    loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=1,
            shuffle=True)

    image, label, name = next(iter(loader))

    print(name)

    f, ax = plt.subplots(2,2,figsize=(8,8))
    ax[0,0].imshow(image[0].data.cpu().permute(1,2,0).numpy())
    ax[0,1].imshow(image[1].data.cpu().permute(1,2,0).numpy())
    ax[1,0].imshow(label[0].data.cpu().numpy())
    ax[1,1].imshow(label[1].data.cpu().numpy())

    print(image.shape, label.shape)
    print(image.max().item(), image.min().item())
    print(image.type(), label.type())
    print(np.unique(label[0].data.cpu().numpy()))
    print(np.unique(label[1].data.cpu().numpy()))
    plt.show()

