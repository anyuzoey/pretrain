
# coding: utf-8

import numpy as np  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class HED(nn.Module):
    def __init__(self):
        super(HED, self).__init__()
        #lr 1 2 decay 1 0
        self.dropout = nn.Dropout(0.2)
        self.conv1_1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv5_1 = nn.Conv2d(256, 256, kernel_size=3,stride=1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3,stride=1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(256, 256, kernel_size=3,stride=1, padding=2, dilation=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)


        #lr 0.1 0.2 decay 1 0
#         self.conv1_1_down = nn.Conv2d(64, 21, 1, padding=0)
        self.conv1_2_down = nn.Conv2d(32, 21, 1, padding=0)

#         self.conv2_1_down = nn.Conv2d(128, 21, 1, padding=0)
        self.conv2_2_down = nn.Conv2d(64, 21, 1, padding=0)

#         self.conv3_1_down = nn.Conv2d(256, 21, 1, padding=0)
#         self.conv3_2_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_3_down = nn.Conv2d(128, 21, 1, padding=0)

#         self.conv4_1_down = nn.Conv2d(512, 21, 1, padding=0)
#         self.conv4_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv4_3_down = nn.Conv2d(256, 21, 1, padding=0)
        
#         self.conv5_1_down = nn.Conv2d(512, 21, 1, padding=0)
#         self.conv5_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv5_3_down = nn.Conv2d(256, 21, 1, padding=0)

        #lr 0.01 0.02 decay 1 0
        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)
        #lr 0.001 0.002 decay 1 0
        self.score_final = nn.Conv2d(5, 1, 1)

    def forward(self, x):
        # VGG
        img_H, img_W = x.shape[2], x.shape[3]
        conv1_1 = self.dropout(self.relu(self.conv1_1(x)))
        conv1_2 = self.dropout(self.relu(self.conv1_2(conv1_1)))
        pool1   = self.maxpool(conv1_2)

        conv2_1 = self.dropout(self.relu(self.conv2_1(pool1)))
        conv2_2 = self.dropout(self.relu(self.conv2_2(conv2_1)))
#         conv2_2 = self.relu(self.conv2_2(pool1))
        pool2   = self.maxpool(conv2_2)

        conv3_1 = self.dropout(self.relu(self.conv3_1(pool2)))
        conv3_2 = self.dropout(self.relu(self.conv3_2(conv3_1)))
        conv3_3 = self.dropout(self.relu(self.conv3_3(conv3_2)))
#         conv3_3 = self.relu(self.conv3_3(pool2))
        pool3   = self.maxpool(conv3_3)

        conv4_1 = self.dropout(self.relu(self.conv4_1(pool3)))
        conv4_2 = self.dropout(self.relu(self.conv4_2(conv4_1)))
        conv4_3 = self.dropout(self.relu(self.conv4_3(conv4_2)))
#         conv4_3 = self.relu(self.conv4_3(pool3))
        pool4   = self.maxpool4(conv4_3)

        conv5_1 = self.dropout(self.relu(self.conv5_1(pool4)))
        conv5_2 = self.dropout(self.relu(self.conv5_2(conv5_1)))
        conv5_3 = self.dropout(self.relu(self.conv5_3(conv5_2)))
#         conv5_3 = self.relu(self.conv5_3(pool4))

#         conv1_1_down = self.conv1_1_down(conv1_1)
        conv1_2_down = self.dropout(self.conv1_2_down(conv1_2))
#         conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.dropout(self.conv2_2_down(conv2_2))
#         conv3_1_down = self.conv3_1_down(conv3_1)
#         conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.dropout(self.conv3_3_down(conv3_3))
#         conv4_1_down = self.conv4_1_down(conv4_1)
#         conv4_2_down = self.conv4_2_down(conv4_2)
        conv4_3_down = self.dropout(self.conv4_3_down(conv4_3))
#         conv5_1_down = self.conv5_1_down(conv5_1)
#         conv5_2_down = self.conv5_2_down(conv5_2)
        conv5_3_down = self.dropout(self.conv5_3_down(conv5_3))
    
        so1_out = self.score_dsn1(conv1_2_down)
        so2_out = self.score_dsn2(conv2_2_down)
        so3_out = self.score_dsn3(conv3_3_down)
        so4_out = self.score_dsn4(conv4_3_down)
        so5_out = self.score_dsn5(conv5_3_down)

#         so1_out = self.score_dsn1(conv1_1_down + conv1_2_down)
#         so2_out = self.score_dsn2(conv2_1_down + conv2_2_down)
#         so3_out = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down)
#         so4_out = self.score_dsn4(conv4_1_down + conv4_2_down + conv4_3_down)
#         so5_out = self.score_dsn5(conv5_1_down + conv5_2_down + conv5_3_down)

        ## transpose and crop way 
        weight_deconv2 =  make_bilinear_weights(4, 1).cuda()
        weight_deconv3 =  make_bilinear_weights(8, 1).cuda()
        weight_deconv4 =  make_bilinear_weights(16, 1).cuda()
        weight_deconv5 =  make_bilinear_weights(32, 1).cuda()

        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, weight_deconv5, stride=8)
        ### center crop
        so1 = crop(so1_out, img_H, img_W)
        so2 = crop(upsample2, img_H, img_W)
        so3 = crop(upsample3, img_H, img_W)
        so4 = crop(upsample4, img_H, img_W)
        so5 = crop(upsample5, img_H, img_W)
        ### crop way suggested by liu 
        # so1 = crop_caffe(0, so1, img_H, img_W)
        # so2 = crop_caffe(1, upsample2, img_H, img_W)
        # so3 = crop_caffe(2, upsample3, img_H, img_W)
        # so4 = crop_caffe(4, upsample4, img_H, img_W)
        # so5 = crop_caffe(8, upsample5, img_H, img_W)
        ## upsample way
        # so1 = F.upsample_bilinear(so1, size=(img_H,img_W))
        # so2 = F.upsample_bilinear(so2, size=(img_H,img_W))
        # so3 = F.upsample_bilinear(so3, size=(img_H,img_W))
        # so4 = F.upsample_bilinear(so4, size=(img_H,img_W))
        # so5 = F.upsample_bilinear(so5, size=(img_H,img_W))

        fusecat = torch.cat((so1, so2, so3, so4, so5), dim=1)
        fuse = self.score_final(fusecat)
        results = [so1, so2, so3, so4, so5, fuse]
        results = [torch.sigmoid(r) for r in results]
        return results
    
def crop(variable, th, tw):
        h, w = variable.shape[2], variable.shape[3]
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return variable[:, :, y1 : y1 + th, x1 : x1 + tw]
def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w

# model = HED()
# model.cuda();
# summary(model, (1, 92, 92))

