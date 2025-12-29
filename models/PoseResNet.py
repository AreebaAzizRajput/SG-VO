# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

# Modified by Yanlin Jin, 2025
# rgb and orb multi-head cross attention, deepest-layer feature


from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .resnet_encoder import *
from einops import rearrange

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, in_channels, emb_dim, num_heads=8, att_dropout=0.0, dropout=0.0):
        super(MultiHeadCrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."

        self.proj_in_q = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)
        self.proj_in_kv = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)
        
        self.att_dropout = nn.Dropout(att_dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, orb, rgb, pad_mask=None):
        '''
        :param rgb: [batch_size, c, h, w] RGB features
        :param orb: [batch_size, c, h, w] orb features
        :param pad_mask: (optional)
        :return:
        '''
        b, c, h, w = rgb.shape
 
        orb = self.proj_in_q(orb)   # [b, 512, 8, 6] => [b, 128, 8, 26]
        orb = rearrange(orb, 'b c h w -> b (h w) c')
        
        rgb = self.proj_in_kv(rgb)
        rgb = rearrange(rgb, 'b c h w -> b (h w) c')


        Q = self.Wq(orb)  # [batch_size, h*w, emb_dim] = [b, 208, 128]
        K = self.Wk(rgb)
        V = self.Wv(rgb)
 
        Q = Q.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, h*w, head_dim]=[b, 8, 208, 16]
        K = K.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2) 
        V = V.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
 
        att_weights = torch.einsum('bnid,bnjd -> bnij', Q, K)
        att_weights = att_weights * self.scale
 
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)
 
        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bnij, bnjd -> bnid', att_weights, V)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.emb_dim)   # [batch_size, h*w, emb_dim]
 
        # print(out.shape)
 
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)   # [batch_size, c, h, w]
        out = self.proj_out(out)

        return out, att_weights


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        pose = 0.01 * out.view(-1, 6)

        return pose


class PoseResNet(nn.Module):

    def __init__(self, num_layers = 18, pretrained = True):
        super(PoseResNet, self).__init__()
        self.encoder_rgb = ResnetEncoder(num_layers = num_layers, pretrained = pretrained, num_input_images=2, input_image_channel=3)
        self.encoder_orb = ResnetEncoder(num_layers = num_layers, pretrained = pretrained, num_input_images=2, input_image_channel=33)
        self.crossAttention = MultiHeadCrossAttention(in_channels=512, emb_dim=128, num_heads=8)
        self.decoder = PoseDecoder(self.encoder_rgb.num_ch_enc, num_input_features=2)

    def init_weights(self):
        pass

    def forward(self, img1, img2, output_weight=False):
        rgb = torch.cat([img1[:,:3,:,:],img2[:,:3,:,:]],1)
        orb = torch.cat([img1[:,3:,:,:],img2[:,3:,:,:]],1)

        features_rgb = self.encoder_rgb(rgb)
        features_orb = self.encoder_orb(orb)
        out, att_weights = self.crossAttention(features_orb[-1], features_rgb[-1]) # deepest features

        # pose = self.decoder([[out + features_rgb[-1]]])
        pose = self.decoder([[features_rgb[-1]],[out]])
        if output_weight:
            return pose, att_weights
        else:
            return pose

if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    model = PoseResNet().cuda()
    model.train()

    tgt_img = torch.randn(4, 36, 256, 832).cuda()
    ref_imgs = [torch.randn(4, 36, 256, 832).cuda() for i in range(2)]

    pose = model(tgt_img, ref_imgs[0])

    print(pose.size())