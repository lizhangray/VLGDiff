import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Concatenate(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)

class GatedDWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, activation=nn.ReLU()):
        super().__init__()
        self.activation = activation
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                   padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(in_channels, out_channels * 2, 1, 1, 0, bias=True)
        self.bn_pw = nn.BatchNorm2d(out_channels * 2)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_dw(x)

        x = self.pointwise(x)
        x = self.bn_pw(x)

        feature, gate = torch.chunk(x, 2, dim=1)

        return self.activation(feature) * torch.sigmoid(gate)

class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


class CASAtt(nn.Module):
    def __init__(self, dim=256, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(SpatialOperation(dim), ChannelOperation(dim))
        self.oper_k = nn.Sequential(SpatialOperation(dim), ChannelOperation(dim))
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = [6, 12, 18]
        for rate in rates:
            modules.append(GatedDWConv(in_channels, out_channels, 3, padding=rate, dilation=rate))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class EdgeNetwork(nn.Module):
    def __init__(self):
        super(EdgeNetwork, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=5, stride=1, padding=2),  
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.enc2 = nn.Sequential(
            GatedDWConv(64, 128, stride=2),
            GatedDWConv(128, 128)
        )

        self.enc3 = nn.Sequential(
            GatedDWConv(128, 256, stride=2),
            GatedDWConv(256, 256)
        )

        self.bridge_conv1 = GatedDWConv(256, 256, dilation=2, padding=2)
        self.bridge_conv2 = GatedDWConv(256, 256, dilation=4, padding=4)

        self.attention = CASAtt(dim=256)

        # Up 1: 1/4 -> 1/2
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1_fusion = GatedDWConv(256 + 128, 128)

        # Up 2: 1/2 -> 1
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec2_fusion = GatedDWConv(128 + 64, 64)

        # Output Head
        self.final = nn.Conv2d(64, 1, kernel_size=5, padding=2)  # 输出 Logits

    def forward(self, x):
        # x: [B, 2, H, W] (Gray + Mask)

        # Encoding
        e1 = self.enc1(x)  # [B, 64, H, W]
        e2 = self.enc2(e1)  # [B, 128, H/2, W/2]
        e3 = self.enc3(e2)  # [B, 256, H/4, W/4]

        # Bottleneck
        b = self.bridge_conv1(e3)
        b = self.bridge_conv2(b)
        b = b + self.attention(b) 

        # Decoding
        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)  
        d1 = self.dec1_fusion(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)  
        d2 = self.dec2_fusion(d2)

        edge_logits = self.final(d2)
        return edge_logits

class CompNetwork(nn.Module):
    def __init__(self):
        super(CompNetwork, self).__init__()

        # --- Encoder ---
        self.enc1 = nn.Sequential(nn.Conv2d(5, 64, 5, 1, 2), nn.BatchNorm2d(64), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU())  # Down 1/2
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU())  # Down 1/4


        self.middle_blk1 = GatedDWConv(256, 256, dilation=2, padding=2)
        self.middle_blk2 = GatedDWConv(256, 256, dilation=4, padding=4)

        self.attention = CASAtt(dim=256)  
        self.aspp = ASPP(256, 256)  

        self.middle_blk3 = GatedDWConv(256, 256, dilation=2, padding=2)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = GatedDWConv(256, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = GatedDWConv(128, 64)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()  
        )

    def forward(self, x):
        # Encoder
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        # Bottleneck (Integrated Improvements)
        x = self.middle_blk1(x)
        x = self.middle_blk2(x)

        x = x + self.attention(x)

        x = self.aspp(x)

        x = self.middle_blk3(x)

        # Decoder
        x = self.up1(x)
        x = self.dec1(x)

        x = self.up2(x)
        x = self.dec2(x)

        out = self.final_conv(x)
        return out

class LocalDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(LocalDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.output_shape = (1024,)
        self.img_c = input_shape[0]
        self.img_h = input_shape[1]
        self.img_w = input_shape[2]
        # input_shape: (None, img_c, img_h, img_w)
        self.conv1 = nn.Conv2d(self.img_c, 64, kernel_size=5, stride=2, padding=2)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.activate1 = nn.ReLU()
        # input_shape: (None, 64, img_h//2, img_w//2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.activate2 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.activate3 = nn.ReLU()
        # input_shape: (None, 256, img_h//8, img_w//8)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.batchnorm4 = nn.BatchNorm2d(512)
        self.activate4 = nn.ReLU()
        # input_shape: (None, 512, img_h//16, img_w//16)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.batchnorm5 = nn.BatchNorm2d(512)
        self.activate5 = nn.ReLU()
        # input_shape: (None, 512, img_h//32, img_w//32)
        in_features = 512 * (self.img_h // 32) * (self.img_w // 32)
        self.flatten6 = Flatten()
        # input_shape: (None, 512 * img_h//32 * img_w//32)
        self.linear6 = nn.Linear(in_features, 1024)
        self.activate6 = nn.ReLU()
        # output_shape: (None, 1024)

    def forward(self, x):
        x = self.batchnorm1(self.activate1(self.conv1(x)))
        x = self.batchnorm2(self.activate2(self.conv2(x)))
        x = self.batchnorm3(self.activate3(self.conv3(x)))
        x = self.batchnorm4(self.activate4(self.conv4(x)))
        x = self.batchnorm5(self.activate5(self.conv5(x)))
        x = self.activate6(self.linear6(self.flatten6(x)))
        return x


class GlobalDiscriminator(nn.Module):
    def __init__(self, input_shape, arc='RelicData'):
        super(GlobalDiscriminator, self).__init__()
        self.arc = arc
        self.input_shape = input_shape
        self.output_shape = (1024,)
        self.img_c = input_shape[0]
        self.img_h = input_shape[1]
        self.img_w = input_shape[2]

        # input_shape: (None, img_c, img_h, img_w)
        self.conv1 = nn.Conv2d(self.img_c, 64, kernel_size=5, stride=2, padding=2)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.activate1 = nn.ReLU()
        # input_shape: (None, 64, img_h//2, img_w//2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.activate2 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.activate3 = nn.ReLU()
        # input_shape: (None, 256, img_h//8, img_w//8)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.batchnorm4 = nn.BatchNorm2d(512)
        self.activate4 = nn.ReLU()
        # input_shape: (None, 512, img_h//16, img_w//16)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.batchnorm5 = nn.BatchNorm2d(512)
        self.activate5 = nn.ReLU()
        # input_shape: (None, 512, img_h//32, img_w//32)
        if arc == 'RelicData':
            in_features = 512 * (self.img_h // 32) * (self.img_w // 32)
            self.flatten6 = Flatten()
            self.linear6 = nn.Linear(in_features, 1024)
            self.activate6 = nn.ReLU()
        elif arc == 'places2':
            self.conv6 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
            self.batchnorm6 = nn.BatchNorm2d(512)
            self.activate6 = nn.ReLU()
            # input_shape (None, 512, img_h//64, img_w//64)
            in_features = 512 * (self.img_h // 64) * (self.img_w // 64)
            self.flatten7 = Flatten()
            self.linear7 = nn.Linear(in_features, 1024)
            self.activate7 = nn.ReLU()
        else:
            raise ValueError('Unsupported architecture \'%s\'.' % self.arc)
        # output_shape: (None, 1024)

    def forward(self, x):
        x = self.batchnorm1(self.activate1(self.conv1(x)))
        x = self.batchnorm2(self.activate2(self.conv2(x)))
        x = self.batchnorm3(self.activate3(self.conv3(x)))
        x = self.batchnorm4(self.activate4(self.conv4(x)))
        x = self.batchnorm5(self.activate5(self.conv5(x)))
        if self.arc == 'RelicData':
            x = self.activate6(self.linear6(self.flatten6(x)))
        elif self.arc == 'places2':
            x = self.batchnorm6(self.activate6(self.conv6(x)))
            x = self.activate7(self.linear7(self.flatten7(x)))
        return x


class Discriminator(nn.Module):
    def __init__(self, local_input_shape, global_input_shape, arc='RelicData'):
        super(Discriminator, self).__init__()
        self.arc = arc
        self.input_shape = [local_input_shape, global_input_shape]
        self.output_shape = (1,)
        self.model_ld = LocalDiscriminator(local_input_shape)
        self.model_gd = GlobalDiscriminator(global_input_shape, arc=arc)
        # input_shape: [(None, 1024), (None, 1024)]
        in_features = self.model_ld.output_shape[-1] + self.model_gd.output_shape[-1]
        self.concat1 = Concatenate(dim=-1)
        # input_shape: (None, 2048)
        self.linear1 = nn.Linear(in_features, 1)
        self.activate1 = nn.Sigmoid()
        # output_shape: (None, 1)

    def forward(self, x):
        x_ld, x_gd = x
        x_ld = self.model_ld(x_ld)
        x_gd = self.model_gd(x_gd)
        out = self.activate1(self.linear1(self.concat1([x_ld, x_gd])))
        return out