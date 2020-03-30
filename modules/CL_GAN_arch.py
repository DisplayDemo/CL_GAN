import torch.nn as nn
import torch.nn.functional as F


class CL_layer(nn.Module):
    def __init__(self):
        super(CL_layer, self).__init__()
        self.CL_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.CL_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.CL_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.CL_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        # x[0]=feature x[1]=condition
        scale = self.CL_scale_conv1(F.leaky_relu(self.CL_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.CL_shift_conv1(F.leaky_relu(self.CL_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift


class ResBlock_CL(nn.Module):
    def __init__(self):
        super(ResBlock_CL, self).__init__()
        self.cl0 = CL_layer()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.cl1 = CL_layer()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        feature = self.cl0(x)
        feature = F.relu(self.conv0(feature), inplace=True)
        feature = self.cl1((feature, x[1]))
        feature = self.conv1(feature)

        return (x[0] + feature, x[1])


class CL_Network(nn.Module):
    def __init__(self):
        super(CL_Network, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, 1, 1)
        cl_net = []
        for i in range(16):
            cl_net.append(ResBlock_CL())
        cl_net.append(CL_layer())
        cl_net.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.cl_net = nn.Sequential(*cl_net)

        self.UpSampling = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

        self.Condition = nn.Sequential(
            nn.Conv2d(8, 128, 4, 4),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 32, 1)
        )

    def forward(self, x):
        cond = self.Condition(x[1])
        fea = self.conv0(x[0])
        res = self.cl_net((fea, cond))
        fea = fea + res
        out = self.UpSampling(fea)
        return out



class VGG_19_D(nn.Module):
    def __init__(self):
        super(VGG_19_D, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.LeakyReLU(0.1,True),

            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, True),
        )

        self.GAN = nn.Sequential(
            nn.Linear(512*6*6,100),
            nn.LeakyReLU(0.1,True),
            nn.Linear(100,1)
        )

        self.input_class = nn.Sequential(
            nn.Linear(512 * 6 * 6, 100),
            nn.LeakyReLU(0.1, True),
            nn.Linear(100, 8)
        )

    def forward(self, *x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0),-1)
        GAN = self.GAN(feature)
        input_class = self.input_class(feature)

        return[GAN,input_class]

