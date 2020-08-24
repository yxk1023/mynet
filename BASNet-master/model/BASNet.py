import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from .resnet_model import *


class SENet(nn.Module):

    def __init__(self, in_dim):
        super(SENet, self).__init__()
        self.dim = in_dim
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, self.dim // 2, 1, 1, 0), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(self.dim // 2, self.dim, 1, 1, 0), nn.ReLU())
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        global_pool = self.global_pool(x)
        conv1 = self.conv1(global_pool)
        conv2 = self.conv2(conv1)
        #conv3 = self.sigmoid(conv2).expand(1, self.dim, 14, 14)
        conv3 = self.sigmoid(conv2).expand_as(x).clone()

        output = torch.mul(x,conv3)

        return output


class SANet(nn.Module):

    def __init__(self, in_dim):
        super(SANet, self).__init__()
        self.dim = in_dim
        self.k = 9
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_dim, self.dim // 2, (1, self.k), 1, (0, self.k // 2)), nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_dim, self.dim // 2, (self.k, 1), 1, (self.k // 2, 0)), nn.ReLU())
        self.conv2_1 = nn.Conv2d(self.dim // 2, 1, (self.k, 1), 1, (self.k // 2, 0))
        self.conv2_2 = nn.Conv2d(self.dim // 2, 1, (1, self.k), 1, (0, self.k // 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(x)
        conv2_1 = self.conv2_1(conv1_1)
        conv2_2 = self.conv2_2(conv1_2)
        conv3 = torch.add(conv2_1, conv2_2)
        conv4 = self.sigmoid(conv3)
        conv5 = conv4.repeat(1, self.dim, 1, 1)

        return conv5


class BASNet(nn.Module):
    def __init__(self,n_channels,n_classes):
        super(BASNet,self).__init__()

        resnet = models.resnet34(pretrained=True)

        ## -------------Encoder--------------

        self.inconv = nn.Conv2d(n_channels,64,3,padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)

        #stage 1
        self.encoder1 = resnet.layer1 #256
        #stage 2
        self.encoder2 = resnet.layer2 #128
        #stage 3
        self.encoder3 = resnet.layer3 #64
        #stage 4
        self.encoder4 = resnet.layer4 #32
        #stage 5
        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)


        self.outside1 = nn.Sequential(
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,128,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(4,4,ceil_mode=True)
        )
        self.outside2 = nn.Sequential(
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2,2,ceil_mode=True)
        )
        self.outside3 = nn.Sequential(
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,128,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.outside4 = nn.Sequential(
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.outside5 = nn.Sequential(
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2,mode='bilinear')
        )
        self.hlow = nn.Sequential(
            nn.Conv2d(384,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.hhigh = nn.Sequential(
            nn.Conv2d(512,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.SE =SENet(512)
        self.h5SE = nn.Sequential(
            nn.Conv2d(512,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.SA =SANet(128)
        self.SA1 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear')
        )
        self.SA0 = nn.Upsample(scale_factor=2,mode='bilinear')

        self.out1= nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(256,1,1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        # ## -------------Refine Module-------------
        # self.refunet = RefUnet(1,64)


    def forward(self,x):

        hx = x

        ## -------------Encoder-------------
        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)

        h1 = self.encoder1(hx) # 256    第一层侧输出
        h2 = self.encoder2(h1) # 128    第二层侧输出
        h3 = self.encoder3(h2) # 64     第三层侧输出
        h4 = self.encoder4(h3) # 32     第四层侧输出
        h5 = self.pool4(h4) # 16        第五层侧输出

        h11 = self.outside1(h1)     #56*56*128
        h21 = self.outside2(h2)     #56*56*128
        h31 = self.outside3(h3)     #56*56*128
        h41 = self.outside4(h4)     #28*28*256
        h51 = self.outside5(h5)     #28*28*256

        hlow = self.hlow(torch.cat(((torch.cat((h11, h21), 1)),h31),1))
        hhigh = self.hhigh(torch.cat((h41,h51),1))

        h5SE = self.SE(h5)  # 14*14*512
        h5SE2 = self.h5SE(h5SE)  # 28*28*128

        SA1 = self.SA(torch.mul(hhigh,h5SE2))
        SA2 = self.SA1(SA1)
        SA0 = self.SA0(torch.mul(hhigh,h5SE2))

        out = self.out1(torch.cat((SA0,torch.mul(hlow,SA2)),1))
        ## -------------Refine Module-------------
        # dout = self.refunet(d1) # 256

        return F.sigmoid(out)
# BASNet = BASNet()
# print(BASNet)