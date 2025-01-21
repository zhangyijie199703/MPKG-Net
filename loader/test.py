import os
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torch
import cv2

print(__file__)

s1 = "data"
s2 = "img.png"

s3 = os.path.join(s1, s2)
print(s3)


# img = cv2.imread('/home/zhangyijie/Desktop/graduation/loader/1.png')
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# def get_ISPRS():
#     return np.asarray(
#         [
#             [255, 255, 255],  # 不透水面
#             [0, 0, 255],  # 建筑物
#             [0, 255, 255],  # 低植被
#             [0, 255, 0],  # 树
#             [255, 255, 0],  # 车
#             [255, 0, 0],  # Clutter/background
#             [0, 0, 0]  # ignore
#         ]
#     )
#
# def encode_segmap(mask):
#     """Encode segmentation label images as pascal classes
#
#     Args:
#         mask (np.ndarray): raw segmentation label image of dimension
#           (M, N, 3), in which the Pascal classes are encoded as colours.
#
#     Returns:
#         (np.ndarray): class map with dimensions (M,N), where the value at
#         a given location is the integer denoting the class index.
#     """
#     mask = mask.astype(int)
#     label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
#
#     for ii, label in enumerate(get_ISPRS()):
#         label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
#     label_mask = label_mask.astype(int)
#     label_mask[label_mask == 6] = 255
#     # plt.imshow(label_mask)
#     # plt.title('Remote Sense')
#     # pylab.show()
#     return label_mask
#
# def Normalize(image, label, div_std=False):
#     image_mean = [0.38083647, 0.33612353, 0.35943412]
#     image_std = [0.10240941, 0.10278902, 0.10292039]
#
#     image = np.array(image).astype(np.float32)
#     # print(image.shape)
#     # print(image)
#     image /= 255
#     image -= image_mean
#     # print(image)
#
#     return image, label
#
#
# def toTensor(image, label):
#     image = np.array(image).astype(np.float32).transpose((2, 0, 1))
#     image = torch.from_numpy(image).type(torch.FloatTensor)
#
#     return image, label
#
root = '/home/zhangyijie/Desktop/graduation/loader/vaismall/train/image'
split = 'train'
image_list = []
label_list = []
# # print(os.path.join(root, split, 'image'))
# path = '/home/zhangyijie/Desktop/graduation/loader/vaismall/train'
# 循环获取文件夹下每个文件的名字,并将图片和对应的label地址装入list
for image_fp in os.listdir(root):
    # print(image_fp)
    # IRRG img path
    image_path = os.path.join(root, split, 'image', image_fp)
    # Label path
    label_fp = image_fp
    label_fp= label_fp.replace("area", "zhangyijie")
    print(label_fp)
    label_path = os.path.join(root, split, 'label', label_fp)

    image_list.append(image_path)
    label_list.append(label_path)
#
# # print(image_list)
# # print(label_list)
#
# image_path = image_list[1]
# label_path = label_list[1]
# # 获取文件的名字，不要名字前的路径
# name = os.path.basename(image_path)
# # print(image_path)
# # print(name)
#
# img = Image.open(image_path)
# label = Image.open(label_path)
#
#
# # print(img.size)
# # print(label.size)
# # print(img)
# # print(label)
#
# image, label = Normalize(img, label)
# image, label = toTensor(image,  label)
#
# print(label)
# label = np.asarray(label, dtype=np.uint8)
# print(label.shape)
# print(label)
# label = encode_segmap(label).astype(np.uint8)  # numpy
# print(label.shape)
# print(label)
#
# # print(image.shape)
# # image = torch.cat([image], dim=0)
# # print(image.shape)
#
# # print(type(image))
# # print(image.dtype)
# # print(image.shape)
# # print(label)
#
#
