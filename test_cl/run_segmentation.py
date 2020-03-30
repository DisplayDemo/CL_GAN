import os
import os.path
import glob
import cv2
import numpy as np

import torch
import torchvision.utils

import modules.segmentation_arch as seg_arch
from script.utils import imresize, modcrop

device_c = 'cpu'
train_image_folder_name = 'train_image'

train_image_folder = '../data/' + train_image_folder_name  # HR image
save_prob_path = '../data/' + train_image_folder_name + '_segprob/'
save_byteimg_path = '../data/' + train_image_folder_name + '_byteimg/'
save_colorimg_path = '../data/' + train_image_folder_name + '_colorimg/'

seg_model = seg_arch.OutdoorSceneSeg()
model_path = '../model/segmentation_OST_bic.pth'
seg_model.load_state_dict(torch.load(model_path), strict=True)
seg_model.eval()
seg_model = seg_model.to(device_c)

lookup_table = torch.from_numpy(
    np.array([
        [153, 153, 153],  # 0, background
        [0, 255, 255],  # 1, sky
        [109, 158, 235],  # 2, water
        [183, 225, 205],  # 3, grass
        [153, 0, 255],  # 4, mountain
        [17, 85, 204],  # 5, building
        [106, 168, 79],  # 6, plant
        [224, 102, 102],  # 7, animal
        [255, 255, 255],  # 8/255, void
    ])).float()
lookup_table /= 255
print('Segmentation testing ...')

index = 0

folder_list = os.listdir(train_image_folder)

for folder in folder_list:
    inner_path = os.path.join(train_image_folder, folder)

    filelist = os.listdir(inner_path)
    print('folder name : ' + folder)
    for item in filelist:
        path = os.path.join(os.path.abspath(inner_path), item)
        index += 1
        basename = os.path.basename(path)
        base = os.path.splitext(basename)[0]
        print(index, " ", base)

        # reading image
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = modcrop(img, 8)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        else:
            img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

        # matlab imersize
        img_LR = imresize(img / 255, 1 / 4, antialiasing=True)
        img = imresize(img_LR, 4, antialiasing=True) * 255

        img[0] -= 103.939
        img[1] -= 116.779
        img[2] -= 123.68

        img = img.unsqueeze(0)
        img = img.to(device_c)
        output = seg_model(img).float().cpu().squeeze_()

        # calculate prob
        if not os.path.exists(os.path.join(save_prob_path, folder)):
            os.makedirs(os.path.join(save_prob_path, folder))
        torch.save(output, os.path.join(save_prob_path, folder, base + '_bic.pth'))

        # byteimg
        if not os.path.exists(os.path.join(save_byteimg_path, folder)):
            os.makedirs(os.path.join(save_byteimg_path, folder))
        _, argmax = torch.max(output, 0)
        argmax = argmax.squeeze().byte()
        cv2.imwrite(os.path.join(save_byteimg_path, folder, base + '.png'), argmax.numpy())

        im_h, im_w = argmax.size()
        color = torch.FloatTensor(3, im_h, im_w).fill_(0)  # black
        for i in range(8):
            mask = torch.eq(argmax, i)
            color.select(0, 0).masked_fill_(mask, lookup_table[i][0])
            color.select(0, 1).masked_fill_(mask, lookup_table[i][1])
            color.select(0, 2).masked_fill_(mask, lookup_table[i][2])

        mask = torch.eq(argmax, 255)
        color.select(0, 0).masked_fill_(mask, lookup_table[8][0])  # R
        color.select(0, 1).masked_fill_(mask, lookup_table[8][1])  # G
        color.select(0, 2).masked_fill_(mask, lookup_table[8][2])  # B
        if not os.path.exists(os.path.join(save_colorimg_path, folder)):
            os.makedirs(os.path.join(save_colorimg_path, folder))
        torchvision.utils.save_image(color, os.path.join(save_colorimg_path, folder, base + '.png'), padding=0,
                                     normalize=False)
