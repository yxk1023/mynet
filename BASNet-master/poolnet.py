import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np

from .deeplab_resnet import resnet50_locate
from .vgg import vgg16_locate


config_vgg = {'convert': [[128,256,512,512,512],[64,128,256,512,512]], 'deep_pool': [[512, 512, 256, 128], [512, 256, 128, 128], [True, True, True, False], [True, True, True, False]], 'score': 128}  # no convert layer, no conv6

config_resnet = {'convert': [[64,256,512,1024,2048],[128,256,256,512,512]], 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False]], 'score': 128}
        

class PoolNet(nn.Module):
    def __init__(self, base_model_cfg, base):
        super(PoolNet, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.base = base
        #self.deep_pool = nn.ModuleList(deep_pool_layers)
        #self.score = score_layers
    
        self.predict_layer6 = global_feature()

        self.fem_layer5 = feature_exctraction(512,64)
        self.fem_layer4 = feature_exctraction(512,64)
        self.fem_layer3 = feature_exctraction(256,64)
        self.fem_layer2 = feature_exctraction(128,64)
        self.fem_layer1 = feature_exctraction(64,64)

        self.conv4 = nn.Conv2d(64,1,3,1,1) 
        self.conv1 = nn.Conv2d(512,1,3,1,1)       

        if self.base_model_cfg == 'resnet':
            self.convert = convert_layers

    def forward(self, x):
        x_size = x.size()
        conv1, conv2, conv3, conv4, conv5 = self.base(x)

        #conv2merge = conv2merge[::-1]
        fem_layer5 = self.fem_layer5(conv5)
        fem_layer4 = self.fem_layer4(conv4)
        fem_layer3 = self.fem_layer3(conv3)
        fem_layer2 = self.fem_layer2(conv2)
        fem_layer1 = self.fem_layer1(conv1)

        #stage 6
        #out6 = self.predict_layer6(conv5)
        out6 = self.conv1(conv5)
        
        predict6 = F.upsample(out6,size=x_size[2:], mode='bilinear')
        predict6_5 = F.upsample(out6,size=fem_layer5.size()[2:], mode='bilinear')
        
        #stage 5
        out5 = torch.add(self.conv4(fem_layer5), predict6_5)
        predict5_4 = F.upsample(out5,size=fem_layer4.size()[2:], mode='bilinear')
        predict5 = F.upsample(out5,size=x_size[2:], mode='bilinear')
  
        
        #stage 4
        out4 = torch.add(self.conv4(fem_layer4),predict5_4)
        predict4_3 = F.upsample(out4,size=fem_layer3.size()[2:], mode='bilinear')
        predict4 = F.upsample(out4,size=x_size[2:], mode='bilinear')

        # stage 3
        out3 = torch.add(self.conv4(fem_layer3), predict4_3)
        predict3_2 = F.upsample(out3,size=fem_layer2.size()[2:], mode='bilinear')
        predict3 = F.upsample(out3,size=x_size[2:], mode='bilinear')

        # stage 2
        out2 = torch.add(self.conv4(fem_layer2), predict3_2)
        predict2_1 = F.upsample(out2,size=fem_layer1.size()[2:], mode='bilinear')
        predict2 = F.upsample(out2,size=x_size[2:], mode='bilinear')

        # stage 1
        out1 = torch.add(self.conv4(fem_layer1), predict2_1)
 #       predict2_1 = F.upsample(out2,size=fem_layer1.size()[2:], mode='bilinear')
        predict1 = F.upsample(out1,size=x_size[2:], mode='bilinear')
        
        return predict1, predict2, predict3, predict4, predict5, predict6
        

def build_model(base_model_cfg='vgg'):
    if base_model_cfg == 'vgg':
        return PoolNet(base_model_cfg, vgg16_locate())
    elif base_model_cfg == 'resnet':
        return PoolNet(base_model_cfg, vgg16_locate())

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()

class feature_exctraction(nn.Module):   
    def __init__(self, in_channel=512, depth=64):
        super(feature_exctraction, self).__init__()

        self.conv_output = nn.Conv2d(in_channel, depth, 1, 1)

    def forward(self, x):
 
        output = self.conv_output(x)

        return output        

        
class SENet(nn.Module):

    def __init__(self, in_dim):
        super(SENet, self).__init__()
        self.dim = in_dim
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, self.dim//2, 1, 1, 0), nn. ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(self.dim//2, self.dim, 1, 1, 0), nn. ReLU())
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):
        global_pool = self.global_pool(x)
        conv1 = self.conv1(global_pool)
        conv2 = self.conv2(conv1)
        conv3 = self.sigmoid(conv2).expand(1, self.dim, x.size(2), x.size(3))
 
        output = x*conv3
        
        return output
        
        
class global_feature(nn.Module):

    def __init__(self, in_channel=512, depth=64):
        super(global_feature, self).__init__()
        self.pool5 = nn.AdaptiveMaxPool2d((1,1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)

        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)

        self.conv_output = nn.Sequential(nn.Conv2d(depth*5, depth, 3, 1, 1),nn. ReLU())

        self.conv1 = nn.Conv2d(depth, 1, 3, 1, 1)
    
    def forward(self, x):
        x_size = x.size()
        
        image_features = self.pool5(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=x_size[2:], mode='bilinear')
 
        atrous_block1 = self.atrous_block1(x)
 
        atrous_block6 = self.atrous_block6(x)
 
        atrous_block12 = self.atrous_block12(x)
 
        atrous_block18 = self.atrous_block18(x)
 
        output = self.conv1(self.conv_output(torch.cat([image_features, atrous_block1, atrous_block6, atrous_block12, atrous_block18], dim=1)))

        return output
        
class SANet(nn.Module):

    def __init__(self, in_dim):
        super(SANet, self).__init__()
        self.dim = in_dim
        self.k = 9
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_dim, self.dim//2, (1, self.k), 1, (0, self.k//2)), nn. ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_dim, self.dim//2, (self.k, 1), 1, (self.k//2, 0)), nn. ReLU())
        self.conv2_1 = nn.Conv2d(self.dim//2, 1, (self.k, 1), 1, (self.k//2, 0))
        self.conv2_2 = nn.Conv2d(self.dim//2, 1, (1, self.k), 1, (0, self.k//2))
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
