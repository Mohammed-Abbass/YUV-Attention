
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import math
from typing import Tuple
import kornia
import torchvision

from collections import OrderedDict
class SKAttention(nn.Module):
    def __init__(self, in_channels,kernels=[3,5],reduction=16,group=1,L=32):
        super().__init__()
        self.d=max(L,in_channels//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(in_channels,in_channels,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(in_channels)), ('relu',nn.ReLU()) ])) )
        self.fc=nn.Linear(in_channels,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,in_channels))
        self.softmax=nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs=[]
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)
        U=sum(conv_outs)
        S=U.mean(-1).mean(-1)
        Z=self.fc(S)
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1))
        attention_weughts=torch.stack(weights,0)
        attention_weughts=self.softmax(attention_weughts)
        V=(attention_weughts*feats).sum(0)
        return V

class QKV(nn.Module):
    def __init__( self, in_dim: int, n_heads: int = 8 ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.qkv_dim = in_dim // n_heads
        self.to_qkv = nn.Linear( in_features=in_dim, out_features=3 * in_dim )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        batch_size, n_tokens, in_dim = input.shape
        qkv = self.to_qkv(input)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.n_heads, self.qkv_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        return qkv.unbind(dim=0)

def get_attention(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    attention = (queries @ keys.transpose(-2, -1)) / math.sqrt(queries.shape[-1])
    attention = F.softmax(attention, dim=-1)
    return attention

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_dim: int =32, n_heads: int = 8) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.to_qkv = QKV( in_dim=in_dim, n_heads=n_heads)
        self.to_output = nn.Linear( in_features=in_dim, out_features=in_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, n_tokens, in_dim,_ = input.shape
        queries, keys, values = self.to_qkv(input)
        attention = get_attention( queries=queries, keys=keys, )
        output = attention @ values

        output = output.transpose(1, 2)
        output = output.reshape(batch_size, n_tokens, in_dim)
        output = self.to_output(output)
        return output

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out

class SEAttention(nn.Module):
    def __init__(self, channel,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential( nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid() )

    def forward(self, x):
        b, c, w, h = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Denoiser(nn.Module):
    def __init__(self, num_filters=32, kernel_size=3):
        super(Denoiser, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels= num_filters, kernel_size=kernel_size, stride=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels= num_filters, kernel_size=kernel_size, stride=1, padding='same')
        self.downPool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.conv3 = nn.Conv2d(in_channels=num_filters, out_channels= num_filters, kernel_size=kernel_size, stride=1, padding='same')
        self.downPool3 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.conv4 = nn.Conv2d(in_channels=num_filters, out_channels= num_filters, kernel_size=kernel_size, stride=1, padding='same')
        self.downPool4 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.bottleneck = TripletAttention()
        self.up2 = nn.MaxUnpool2d(2, stride=2)
        self.up3 = nn.MaxUnpool2d(2, stride=2)
        self.up4 = nn.MaxUnpool2d(2, stride=2)

        self.res_layer = nn.Conv2d(in_channels=num_filters, out_channels= 1, kernel_size=kernel_size, padding='same')
        self.output_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding='same')

    def forward(self, inputs):
        x1 = nn.ReLU()(self.conv1(inputs))
        x2 = nn.ReLU()(self.conv2(x1))
        x2, x2_indices =  self.downPool2(x2)
        x3 = nn.ReLU()(self.conv3(x2))
        x3, x3_indices = self.downPool3(x3)
        x4 = nn.ReLU()(self.conv4(x3))
        x4, x4_indices = self.downPool4(x4)
        x = self.bottleneck(x4)
        x = self.up4(x, x4_indices, output_size=x3.size())
        x = self.up3(x3 + x, x3_indices, output_size=x2.size())
        x = self.up2(x2 + x, x2_indices, output_size=x1.size())
        x = x + x1
        x = nn.Tanh()(self.res_layer(x))
        return nn.Tanh()(self.output_layer(x + inputs))
class LYT_Modified(nn.Module):
    def __init__(self, num_filters=32, kernel_size=3):
        super(LYT_Modified, self).__init__()

        self.process_y = nn.Conv2d(in_channels=1, out_channels= num_filters, kernel_size=kernel_size, stride=1, padding='same')
        self.process_cb = nn.Conv2d(in_channels=1, out_channels= num_filters, kernel_size=kernel_size, stride=1, padding='same')
        self.process_cr = nn.Conv2d(in_channels=1, out_channels= num_filters, kernel_size=kernel_size, stride=1, padding='same')

        self.denoiser_cb = Denoiser()
        self.denoiser_cr = Denoiser()

        self.lum_pool = nn.MaxPool2d(8)
        self.lum_mhsa = TripletAttention()
        self.lum_up = nn.Upsample(scale_factor=8)
        self.lum_conv = nn.Conv2d(in_channels=num_filters, out_channels= num_filters, kernel_size=1, stride=1, padding='same')
        self.ref_conv = nn.Conv2d(in_channels=num_filters*2, out_channels= num_filters, kernel_size=1, stride=1, padding='same')

        self.msef = SEAttention(num_filters)

        self.recombine = nn.Conv2d(in_channels=num_filters*2, out_channels= num_filters, kernel_size=3, stride=1, padding='same')

        self.SKAtten = SKAttention(in_channels=num_filters)

        self.final_adjustments = nn.Conv2d(in_channels=num_filters, out_channels= 3, kernel_size=3, stride=1, padding='same')

    def forward(self, inputs):
        ycbcr = kornia.color.rgb_to_ycbcr(inputs)
        y, cr, cb = ycbcr.chunk(dim=-3, chunks=3)
        cb = self.denoiser_cb(cb) + cb
        cr = self.denoiser_cr(cr) + cr

        y_processed = nn.ReLU()(self.process_y(y))
        cb_processed = nn.ReLU()(self.process_cb(cb))
        cr_processed = nn.ReLU()(self.process_cr(cr))

        ref = torch.concat([cb_processed, cr_processed], dim=1)

        lum = y_processed
        lum_1 = self.lum_pool(lum)
        lum_1 = self.lum_mhsa(lum_1)
        lum_1 = self.lum_up(lum_1)

        m = nn.ConstantPad2d(int(abs(lum_1.shape[2] - lum.shape[2]) / 2), 0)
        lum_1 = m(lum_1)
        lum = lum + lum_1
        ref = self.ref_conv(ref)
        shortcut = ref
        ref = ref + 0.2 * self.lum_conv(lum)
        ref = self.msef(ref)
        ref = ref + shortcut

        recombined = nn.ReLU()(self.recombine(torch.concat([ref, lum], dim=1)))
        output = self.SKAtten(recombined)
        output = self.final_adjustments(output)
        output = nn.Tanh()(output)
        return output

if __name__ == "__main__":
    input = torch.randn(1, 3, 224, 224, requires_grad=False)
    network = LYT_Modified()
    output = network(input)
    print(output.shape)
