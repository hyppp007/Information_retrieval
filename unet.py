# sub-parts of the U-Net model

import torch
import torch.nn as nn
from torch.nn import init
import numpy as np

def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
    if useBN:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU()
        )


def upsample(ch_coarse, ch_fine):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
        nn.ReLU()
    )


class unet(nn.Module):
    def __init__(self, useBN=True):
        super(unet, self).__init__()
        # Downgrade stages
        self.conv1 = add_conv_stage(3, 32, useBN=useBN)
        self.conv2 = add_conv_stage(32, 64, useBN=useBN)
        self.conv3 = add_conv_stage(64, 128, useBN=useBN)
        self.conv4 = add_conv_stage(128, 256, useBN=useBN)
        # self.conv5   = add_conv_stage(256, 512, useBN=useBN)
        # self.conv6   = add_conv_stage(512, 1024, useBN=useBN)

        # Upgrade tages
        # self.conv5m = add_conv_stage(1024, 512, useBN=useBN)
        self.conv4m = add_conv_stage(512, 256, useBN=useBN)
        self.conv3m = add_conv_stage(256, 128, useBN=useBN)
        self.conv2m = add_conv_stage(128, 64, useBN=useBN)
        self.conv1m = add_conv_stage(64, 32, useBN=useBN)
        # Maxpool
        self.max_pool = nn.MaxPool2d(2)
        # Upsample layers
        # self.upsample65 = upsample(1024, 512)
        self.upsample54 = upsample(512, 256)
        self.upsample43 = upsample(256, 128)
        self.upsample32 = upsample(128, 64)
        self.upsample21 = upsample(64, 32)

        self.out_layer = nn.Conv2d(32, 8, 3, 1, 1)  # for 8 classes
        self.softmax = nn.Softmax(dim=1)  # for classification

        ## weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                # kaiming init
                # init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                # init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(self.max_pool(conv1_out))
        conv3_out = self.conv3(self.max_pool(conv2_out))
        conv4_out = self.conv4(self.max_pool(conv3_out))

        conv4m_out_ = torch.cat((self.upsample43(conv4_out), conv3_out), 1)
        conv3m_out = self.conv3m(conv4m_out_)

        conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
        conv2m_out = self.conv2m(conv3m_out_)

        conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
        conv1m_out = self.conv1m(conv2m_out_)

        ## Design your last layer & activations
        out = self.out_layer(conv1m_out)
        out = self.softmax(out)
        return out

