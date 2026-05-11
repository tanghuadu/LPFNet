import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torch.fft


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output

class ALDBModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv1x1_in = Conv(nIn, nIn // 2, 1, 1, padding=0, bn_acti=False)
        self.conv3x1 = Conv(nIn // 2, nIn // 2, (kSize, 1), 1, padding=(1, 0), bn_acti=True)
        self.conv1x3 = Conv(nIn // 2, nIn // 2, (1, kSize), 1, padding=(0, 1), bn_acti=True)

        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.ca22 = AWSA(nIn // 2)

        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)
        self.shuffle = ShuffleBlock(nIn // 2)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv1x1_in(output)
        output = self.conv3x1(output)
        output = self.conv1x3(output)

        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        br2 = self.ca22(br2)

        output = br2 + output
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)
        output = self.shuffle(output + input)

        return output

class AWSA(nn.Module): #11
    """
    Adaptive weighted spatial attention
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.local_conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        b, c, h, w = x.size()

        x_pool = x.mean(dim=2, keepdim=True)
        y_pool = x.mean(dim=3, keepdim=True)

        x_expand = x_pool.expand(-1, -1, h, -1)
        y_expand = y_pool.expand(-1, -1, -1, w)

        sa = (self.alpha * (x_expand * y_expand) + self.beta * (x_expand + y_expand)) / (self.alpha + self.beta + 1e-6)
        sa = self.sigmoid(sa)

        sa = self.local_conv(sa.mean(dim=1, keepdim=True))
        sa = self.sigmoid(sa)
        sa = sa.expand(-1, c, -1, -1)

        out = x * sa
        return out

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
    
class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output

class UpsampleingBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output
        
class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):
        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)
        return out


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class LongConnection(nn.Module):
    def __init__(self, nIn, nOut, kSize, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti
        self.dconv3x1 = nn.Conv2d(nIn, nIn // 2, (kSize, 1), 1, padding=(1, 0))
        self.dconv1x3 = nn.Conv2d(nIn // 2, nOut, (1, kSize), 1, padding=(0, 1))
        
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.dconv3x1(input)
        output = self.dconv1x3(output)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output

#---------------------------------------
# LiteDPG block
#---------------------------------------
class LiteDPGFusion_BN(nn.Module):
    def __init__(self, in_ch, mid_ch, use_add=True, use_mul=True):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.use_add = use_add
        self.use_mul = use_mul

        if use_add:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, in_ch, 1, bias=True)
            )
        else:
            self.channel_add_conv = None

        if use_mul:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, in_ch, 1, bias=True)
            )
        else:
            self.channel_mul_conv = None

    def forward(self, x):
        context = self.avg_pool(x)
        out = x
        if self.use_mul:
            scale = torch.sigmoid(self.channel_mul_conv(context))
            out = out * scale
        if self.use_add:
            bias = self.channel_add_conv(context)
            out = out + bias
        return out

class LiteDPGBlock(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ratio=0.5, use_add=True, use_mul=True):
        super().__init__()
        mid_ch = max(int(in_ch * mid_ratio), out_ch)

        self.reduce = Conv(in_ch, mid_ch, 1, 1, 0,bn_acti=True)

        self.depthwise = Conv(mid_ch, mid_ch, 3, 1, 1, groups=mid_ch, bn_acti=True)

        self.eca = eca_layer(mid_ch, k_size=3)

        self.fusion = LiteDPGFusion_BN(mid_ch, mid_ch // 2, use_add=use_add, use_mul=use_mul)

        self.expand = Conv(mid_ch, out_ch, 1, 1, 0,bn_acti=True)

        self.shortcut = None if in_ch == out_ch else Conv(in_ch, out_ch, 1, 1, 0, bn_acti=False)

    def forward(self, x):
        identity = x
        out = self.reduce(x)
        out = self.depthwise(out)
        out = self.eca(out)
        out = self.fusion(out)
        out = self.expand(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out = out + identity
        return out

class Neck(nn.Module):
    def __init__(self, in_channels_list=[64, 128, 32], out_channels=32):
        super().__init__()
        c1, c2, c3 = in_channels_list
        
        # FPN
        self.lateral_conv3 = Conv(c3, out_channels, 1, 1, padding=0, bn_acti=True)

        self.lateral_conv2 = Conv(c2, out_channels, 1, 1, padding=0, bn_acti=True)
        self.fpn_csp2 = LiteDPGBlock(out_channels * 2, out_channels)

        self.lateral_conv1 = Conv(c1, out_channels, 1, 1, padding=0, bn_acti=True)
        self.fpn_csp1 = LiteDPGBlock(out_channels * 2, out_channels)
        
        # PAN
        self.downsample1 = Conv(out_channels, out_channels, 3, 2, padding=1, bn_acti=True)
        self.pan_csp2 = LiteDPGBlock(out_channels * 2, out_channels)
        
        self.downsample2 = Conv(out_channels, out_channels, 3, 2, padding=1, bn_acti=True)
        self.pan_csp3 = LiteDPGBlock(out_channels * 2, out_channels)
    
    def forward(self, features):
        feat1, feat2, feat3 = features
        
        # FPN
        p3 = self.lateral_conv3(feat3)

        p3_up = F.interpolate(p3, size=feat2.shape[2:], mode='nearest')
        p2_lateral = self.lateral_conv2(feat2)
        p2 = torch.cat([p2_lateral, p3_up], dim=1)
        p2 = self.fpn_csp2(p2)

        p2_up = F.interpolate(p2, size=feat1.shape[2:], mode='nearest')
        p1_lateral = self.lateral_conv1(feat1)
        p1 = torch.cat([p1_lateral, p2_up], dim=1)
        p1 = self.fpn_csp1(p1)
        
        # PAN
        p1_down = self.downsample1(p1)
        n2 = torch.cat([p1_down, p2], dim=1)
        n2 = self.pan_csp2(n2)

        n2_down = self.downsample2(n2)
        n3 = torch.cat([n2_down, p3], dim=1)
        n3 = self.pan_csp3(n3)

        return [p1, n2, n3]

class HierarchicalFusion(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()

        self.global_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

        self.fuse_local = Conv(in_channels * 2, in_channels, 3, 1, 1, bn_acti=True)

    def forward(self, multi_scale_feats, target_size):
        p1, n2, n3 = multi_scale_feats

        w = self.global_fc(n3)
        n2_weighted = n2 * w

        p1_r = F.interpolate(p1, size=target_size, mode='bilinear', align_corners=False)
        n2_r = F.interpolate(n2_weighted, size=target_size, mode='bilinear', align_corners=False)

        fused = self.fuse_local(torch.cat([p1_r, n2_r], dim=1))

        return fused

class LPFNet(nn.Module):
    def __init__(self, classes=19, block_1=3, block_2=12, block_3=12, block_4=3, block_5=3, block_6=3):
        super().__init__()

        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 2, padding=1, bn_acti=True),
        )
        self.bn_prelu_1 = BNPReLU(32)

        self.downsample_1 = DownSamplingBlock(32, 64)
        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), ALDBModule(64, d=2))
        self.bn_prelu_2 = BNPReLU(64)

        dilation_block_2 = [1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32]
        self.downsample_2 = DownSamplingBlock(64, 128)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i),
                                        ALDBModule(128, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(128)

        dilation_block_3 = [1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32]
        self.downsample_3 = DownSamplingBlock(128, 32)
        self.DAB_Block_3 = nn.Sequential()
        for i in range(0, block_3):
            self.DAB_Block_3.add_module("DAB_Module_3_" + str(i),
                                        ALDBModule(32, d=dilation_block_3[i]))
        self.bn_prelu_4 = BNPReLU(32)

        self.neck = Neck(in_channels_list=[64, 128, 32], out_channels=32)

        self.fusion = HierarchicalFusion(in_channels=32)

        dilation_block_4 = [2, 2, 2]
        self.DAB_Block_4 = nn.Sequential()
        for i in range(0, block_4):
           self.DAB_Block_4.add_module("DAB_Module_4_" + str(i),
                                       ALDBModule(32, d=dilation_block_4[i]))
        self.upsample_1 = UpsampleingBlock(32, 16)
        self.bn_prelu_5 = BNPReLU(16)

        dilation_block_5 = [2, 2, 2]
        self.DAB_Block_5 = nn.Sequential()
        for i in range(0, block_5):
            self.DAB_Block_5.add_module("DAB_Module_5_" + str(i),
                                        ALDBModule(16, d=dilation_block_5[i]))
        self.upsample_2 = UpsampleingBlock(16, 16)
        self.bn_prelu_6 = BNPReLU(16)

        dilation_block_6 = [2, 2, 2]
        self.DAB_Block_6 = nn.Sequential()
        for i in range(0, block_6):
            self.DAB_Block_6.add_module("DAB_Module_6_" + str(i),
                                        ALDBModule(16, d=dilation_block_6[i]))
        self.upsample_3 = UpsampleingBlock(16, 16)
        self.bn_prelu_7 = BNPReLU(16)

        self.PA3 = PA(16)

        self.LC1 = LongConnection(64, 16, 3)
        self.LC2 = LongConnection(128, 16, 3)
        self.LC3 = LongConnection(32, 32, 3)

        self.classifier = nn.Sequential(Conv(16, classes, 1, 1, padding=0))

    def forward(self, input):
        output0 = self.init_conv(input)
        output0 = self.bn_prelu_1(output0)

        output1_0 = self.downsample_1(output0)
        output1 = self.DAB_Block_1(output1_0)
        output1 = self.bn_prelu_2(output1)

        output2_0 = self.downsample_2(output1)
        output2 = self.DAB_Block_2(output2_0)
        output2 = self.bn_prelu_3(output2)

        output3_0 = self.downsample_3(output2)
        output3 = self.DAB_Block_3(output3_0)
        output3 = self.bn_prelu_4(output3)
        multi_scale_feats = self.neck([output1, output2, output3])

        neck_output = self.fusion(multi_scale_feats, target_size=output3.shape[2:])

        output4 = self.DAB_Block_4(neck_output)
        output4 = self.upsample_1(output4 + self.LC3(output3))
        output4 = self.bn_prelu_5(output4)

        output5 = self.DAB_Block_5(output4)
        output5 = self.upsample_2(output5 + self.LC2(output2))
        output5 = self.bn_prelu_6(output5)

        output6 = self.DAB_Block_6(output5)
        output6 = self.upsample_3(output6 + self.LC1(output1))
        output6 = self.PA3(output6)
        output6 = self.bn_prelu_7(output6)

        out = F.interpolate(output6, input.size()[2:], mode='bilinear', align_corners=False)
        out = self.classifier(out)
        return out

"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LPFNet(classes=19).to(device)
    summary(model, (1, 3, 512, 1024))