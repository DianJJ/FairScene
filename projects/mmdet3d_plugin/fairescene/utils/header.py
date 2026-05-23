
import torch.nn as nn
import torch
import torch.nn.functional as F
# class Header(nn.Module):
#     def __init__(
#         self,
#         class_num,
#         feature,
#     ):
#         super(Header, self).__init__()
#         self.feature = feature
#         self.class_num = class_num
#         # self.mlp_head = nn.Sequential(
#         #     nn.LayerNorm(self.feature),
#         #     #nn.Linear(self.feature, self.class_num),
#         #     nn.Linear(self.feature, 1),
#         # )
#         self.convmlp_head = ConvMLPHead(self.feature,self.class_num)
#         self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
#         ################################
#         # self.group_conv = nn.Conv3d(self.feature, self.class_num, kernel_size=1, padding=1, groups=20)
#         # self.relu = nn.ReLU()
#         # self.fc = nn.Conv3d(self.class_num, self.class_num, kernel_size=1)
#
#     def forward(self, x3d_l1):
#         # [1, 64, 128, 128, 16]
#
#
#         x3d_up_l1 = self.up_scale_2(x3d_l1) # [1, dim, 128, 128, 16] -> [1, dim, 256, 256, 32]
#
#         _, feat_dim, w, l, h  = x3d_up_l1.shape
#
#         ##################
#         # x = self.group_conv(x3d_up_l1)
#         # x = self.relu(x)
#         # x = self.fc(x)
#         ####################
#         # x3d_up_l1 = x3d_up_l1.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)
#         # # shape = x3d_up_l1.shape
#         # # pdb.set_trace()
#         ##################
#         ssc_logit_full = self.convmlp_head(x3d_up_l1)
#         #ssc_logit_full = x3d_up_l1[:,:20,:,:,:]
#         #
#         # #res["ssc_logit"] = ssc_logit_full.reshape(w, l, h, self.class_num).permute(3,0,1,2).unsqueeze(0)
#         #res = ssc_logit_full.reshape(w, l, h, 1).permute(3, 0, 1, 2).unsqueeze(0)
#         ######################
#         res = ssc_logit_full
#         #[1,1,256,256,32]
#         #pdb.set_trace()
#         return res
#
#
# class SparseHeader(nn.Module):
#     def __init__(self, class_num, feature):
#         super().__init__()
#
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(feature),
#             nn.Linear(feature, class_num)
#         )
#
#     def forward(self, x):
#         x = self.mlp_head(x)
#
#         return x
#
#
#
# class ConvMLPHead(nn.Module):
#     def __init__(self, feature, class_num=20):
#         super(ConvMLPHead, self).__init__()
#         self.layer_norm = nn.LayerNorm(feature)
#         self.feature = feature
#         self.class_num = class_num
#         self.conv = nn.Conv3d(self.feature, self.class_num, kernel_size=1, groups=20)
#
#     def forward(self, x):
#         # x shape: [batch, channels, height, width, lenth]
#         #
#         # x = x.permute(0, 2, 3, 4, 1)  # [batch, height, width, channels]
#         # x = self.layer_norm(x)
#         x = self.conv(x)
#         #x = x[:,:20,:,:,:]
#         return x


import torch.nn as nn


class Header(nn.Module):
    def __init__(
            self,
            class_num,
            feature,
    ):
        super(Header, self).__init__()
        self.feature = feature
        self.class_num = class_num
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature),
            nn.Linear(self.feature, self.class_num),
        )

        self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x3d_l1):
        # [1, 64, 128, 128, 16]
        res = {}

        x3d_up_l1 = self.up_scale_2(x3d_l1)  # [1, dim, 128, 128, 16] -> [1, dim, 256, 256, 32]

        _, feat_dim, w, l, h = x3d_up_l1.shape

        x3d_up_l1 = x3d_up_l1.squeeze().permute(1, 2, 3, 0).reshape(-1, feat_dim)
        # shape = x3d_up_l1.shape
        ssc_logit_full = self.mlp_head(x3d_up_l1)

        #res["ssc_logit"] = ssc_logit_full.reshape(w, l, h, self.class_num).permute(3, 0, 1, 2).unsqueeze(0)
        res["ssc_logit"] = ssc_logit_full.reshape(w, l, h, self.class_num).unsqueeze(0)
        return res


class SparseHeader(nn.Module):
    def __init__(self, class_num, feature):
        super().__init__()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(feature),
            nn.Linear(feature, class_num)
        )

    def forward(self, x):
        x = self.mlp_head(x)

        return x


class SegNet(nn.Module):
    def __init__(self, in_channels=160, num_classes=20, groups=20):
        super(SegNet, self).__init__()

        # Ensure that in_channels is divisible by groups
        assert in_channels % groups == 0, "in_channels must be divisible by groups"

        # Encoder path
        self.enc1 = self.double_conv(in_channels, 160, groups)  # Adjust channels to 160
        self.enc2 = self.double_conv(160, 320, groups)
        self.enc3 = self.double_conv(320, 640, groups)
        self.enc4 = self.double_conv(640, 960, groups)

        # Bottleneck (deepest layer)
        self.bottleneck = self.double_conv(960, 1280, groups)

        # Decoder path
        self.upconv4 = self.up_conv(1280, 960, groups)
        self.dec4 = self.double_conv(960 + 960, 960, groups)

        self.upconv3 = self.up_conv(960, 640, groups)
        self.dec3 = self.double_conv(640 + 640, 640, groups)

        self.upconv2 = self.up_conv(640, 320, groups)
        self.dec2 = self.double_conv(320 + 320, 320, groups)

        self.upconv1 = self.up_conv(320, 160, groups)
        self.dec1 = self.double_conv(160 + 160, 160, groups)

        # Final output layer
        self.final_conv = nn.Conv2d(160, num_classes, kernel_size=1)

    def double_conv(self, in_channels, out_channels, groups):
        """(group convolution => [GroupNorm] => ReLU) * 2"""
        assert in_channels % groups == 0 and out_channels % groups == 0, \
            "in_channels and out_channels must be divisible by groups"
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=groups),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_channels, out_channels, groups):
        """Up-sampling followed by group convolution"""
        assert in_channels % groups == 0 and out_channels % groups == 0, \
            "in_channels and out_channels must be divisible by groups"
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, groups=groups)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)  # Input size (160 channels)
        e2 = self.enc2(F.max_pool2d(e1, 2))  # Output: 320 channels
        e3 = self.enc3(F.max_pool2d(e2, 2))  # Output: 640 channels
        e4 = self.enc4(F.max_pool2d(e3, 2))  # Output: 960 channels

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))  # Output: 1280 channels

        # Decoder path
        d4 = self.upconv4(b)
        d4 = F.interpolate(d4, size=e4.shape[2:])  # Adjust spatial dimensions to match e4
        d4 = torch.cat((d4, e4), dim=1)  # Concatenate with encoder feature (1920 channels)
        d4 = self.dec4(d4)  # Output: 960 channels

        d3 = self.upconv3(d4)
        d3 = F.interpolate(d3, size=e3.shape[2:])  # Adjust spatial dimensions to match e3
        d3 = torch.cat((d3, e3), dim=1)  # Concatenate with encoder feature (1280 channels)
        d3 = self.dec3(d3)  # Output: 640 channels

        d2 = self.upconv2(d3)
        d2 = F.interpolate(d2, size=e2.shape[2:])  # Adjust spatial dimensions to match e2
        d2 = torch.cat((d2, e2), dim=1)  # Concatenate with encoder feature (640 channels)
        d2 = self.dec2(d2)  # Output: 320 channels

        d1 = self.upconv1(d2)
        d1 = F.interpolate(d1, size=e1.shape[2:])  # Adjust spatial dimensions to match e1
        d1 = torch.cat((d1, e1), dim=1)  # Concatenate with encoder feature (320 channels)
        d1 = self.dec1(d1)  # Output: 160 channels

        # Final Group Convolution for 20 classes
        out = self.final_conv(d1)
        #out = torch.sigmoid(out)
        #out = torch.softmax(out)
        return out


