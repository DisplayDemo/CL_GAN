import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import script.utils as util


class LRHR_dataset(data.Dataset):
    def __init__(self, opt):
        super(LRHR_dataset, self).__init__()
        self.opt = opt
        self.LR_paths = None
        self.HR_paths = None
        self.HR_BG_paths = None

        self.LR_paths = util.get_image_paths(opt['data_LR'])
        self.HR_paths = util.get_image_paths(opt['data_HR'])
        self.HR_BG_paths = util.get_image_paths(opt['data_HR_BG'])

        assert self.HR_paths, 'Error HR path is empty'
        if self.LR_paths and self.HR_paths:
            assert len(self.LR_paths) == len(self.HR_paths), \
                'HR and LR datasets have different number of images - {}, {}.'.format( \
                    len(self.LR_paths), len(self.HR_paths))

        self.random_scale_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5]
        self.ratio = 10  # random sampling DIV2K generate image to expand data sets

    def __getitem__(self, item):
        HR_path, LR_path = None, None
        scale = self.opt['scale']
        HR_size = self.opt['HR_size']

        if self.opt['phase'] == 'train' and random.choice(list(range(self.ratio))) == 0:
            bg_index = random.randint(0, len(self.HR_BG_paths) - 1)
            HR_path = self.HR_BG_paths[bg_index]
            img_HR = util.read_img(HR_path)
            seg = torch.FloatTensor(8, img_HR.shape[0], img_HR.shape[1]).fill_(0)
            seg[0, :, :] = 1

        else:
            HR_path = self.HR_paths[item]
            img_HR = util.read_img(HR_path)
            seg = torch.load(HR_path.replace('/train_image/', '/train_image_segprob/').replace('.png', '_bic.pth'))

        if self.opt['phase'] != 'train':
            img_HR = util.modcrop(img_HR, 8)

        seg = np.transpose(seg.detach().numpy(), (1, 2, 0))

        if self.LR_paths:
            LR_path = self.LR_paths[item]
            img_LR = util.read_img(LR_path)
        else:
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = seg.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, HR_size)
                W_s = _mod(W_s, random_scale, scale, HR_size)
                img_HR = cv2.resize(np.copy(img_HR), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                seg = cv2.resize(np.copy(seg), (W_s, H_s), interpolation=cv2.INTER_NEAREST)

            H, W, _ = img_HR.shape

            img_LR = util.imresize(img_HR, 1 / scale, True)
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)

        H, W, C = img_LR.shape
        if self.opt['phase'] == 'train':
            LR_size = HR_size // scale

            # randomly crop
            rand_h = random.randint(0, max(0, H - LR_size))
            rand_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rand_h:rand_h + LR_size, rand_w:rand_w + LR_size, :]
            rand_h_HR, rand_w_HR = int(rand_h * scale), int(rand_w * scale)
            img_HR = img_HR[rand_h_HR:rand_h_HR + HR_size, rand_w_HR:rand_w_HR + HR_size, :]

            # 旋转，翻转
            img_LR, img_HR, seg = util.augment([img_LR, img_HR, seg], self.opt['use_flip'],
                                               self.opt['use_rot'])

            # category
            if 'building' in HR_path:
                category = 1
            elif 'plant' in HR_path:
                category = 2
            elif 'mountain' in HR_path:
                category = 3
            elif 'water' in HR_path:
                category = 4
            elif 'sky' in HR_path:
                category = 5
            elif 'grass' in HR_path:
                category = 6
            elif 'animal' in HR_path:
                category = 7
            else:
                category = 0  # background

        else:
            category = -1  # during val, useless

        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()
        seg = torch.from_numpy(np.ascontiguousarray(np.transpose(seg, (2, 0, 1)))).float()

        if LR_path is None:
            LR_path = '../data/train_LR/'

        return {
            'LR': img_LR,
            'HR': img_HR,
            'seg': seg,
            'category': category,
            'LR_path': LR_path,
            'HR_path': HR_path
        }

    def __len__(self):
        return len(self.HR_paths)
