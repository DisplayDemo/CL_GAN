import os.path
import glob
import cv2
import numpy as np
import torch
from script.utils import imresize, modcrop, tensor2img, save_img
import modules.CL_GAN_arch as cl

device_c = 'cpu'

model_path = '../model/cl_gan.pth'
test_image_path_name = 'test_image'
test_image_path = '../data/' + test_image_path_name
test_prob_path = '../data/' + test_image_path_name + '_segprob'
save_result_path = '../data/' + test_image_path_name + '_result'

model = cl.CL_Network()
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device_c)

print('CL_GAN is Testing ...')

index = 0
for path in glob.glob(test_image_path + '/*'):
    index += 1
    basename = os.path.basename(path)
    base = os.path.splitext(basename)[0]
    print(index, ' ', base)

    # read image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = modcrop(img, 8)
    img = img * 1.0 / 255
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()

    # image imresize
    img_LR = imresize(img, 1 / 4, antialiasing=True)
    img_LR = img_LR.unsqueeze(0)
    img_LR = img_LR.to('cpu')

    seg = torch.load(os.path.join(test_prob_path, base + "_bic.pth"))
    seg = seg.unsqueeze(0)

    seg = seg.to(device_c)

    out = model((img_LR, seg)).data
    out = tensor2img(out.squeeze())
    save_img(out, os.path.join(save_result_path, base + '_rlt.png'))
