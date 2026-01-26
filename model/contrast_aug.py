'''
* Borrow from https://github.com/cheng-01037/Causality-Medical-Image-Domain-Generalization/blob/main/models/imagefilter.py
'''

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from skimage.exposure import equalize_hist
from utils.Functions import imgnorm_torch


class GradlessGCReplayNonlinBlock(nn.Module):
    def __init__(self, out_channel=32, in_channel=3, scale_pool=None, layer_id=0, use_act=True, requires_grad=False,
                 **kwargs):
        """
        Conv-leaky relu layer. Efficient implementation by using group convolutions
        """
        super(GradlessGCReplayNonlinBlock, self).__init__()
        if scale_pool is None:
            scale_pool = [1]
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.scale_pool = scale_pool
        self.layer_id = layer_id
        self.use_act = use_act
        self.requires_grad = requires_grad
        assert requires_grad == False

    def forward(self, x_in, requires_grad=False):
        """
        Args:
            x_in: [ nb (original), nc (original), nx, ny ]
        """
        # random size of kernel
        idx_k = torch.randint(high=len(self.scale_pool), size=(1,))
        k = self.scale_pool[idx_k[0]]

        nb, nc, nx, ny, nz = x_in.shape

        ker = torch.rand([self.out_channel * nb, self.in_channel, k, k, k], requires_grad=self.requires_grad).cuda()
        ker = ker * 2.0 - 1.0
        shift = torch.randn([self.out_channel * nb, 1, 1, 1], requires_grad=self.requires_grad).cuda() * 1.0

        x_in = x_in.view(1, nb * nc, nx, ny, nz)
        x_conv = F.conv3d(x_in, ker, stride=1, padding=k // 2, dilation=1, groups=nb)
        x_conv = x_conv + shift
        if self.use_act:
            x_conv = F.leaky_relu(x_conv)

        x_conv = x_conv.view(nb, self.out_channel, nx, ny, nz)
        return x_conv


class GINGroupConv(nn.Module):
    def __init__(self, out_channel=1, in_channel=3, interm_channel=8, scale_pool=None, n_layer=4, **kwargs):
        '''
        GIN
        '''
        super(GINGroupConv, self).__init__()
        if scale_pool is None:
            scale_pool = [1]
        self.scale_pool = scale_pool  # don't make it tool large as we have multiple layers
        self.n_layer = n_layer
        self.layers = []
        self.out_channel = out_channel
        self.in_channel = in_channel

        self.layers.append(
            GradlessGCReplayNonlinBlock(out_channel=interm_channel, in_channel=in_channel, scale_pool=scale_pool,
                                        layer_id=0).cuda()
        )
        for ii in range(n_layer - 2):
            self.layers.append(
                GradlessGCReplayNonlinBlock(out_channel=interm_channel, in_channel=interm_channel,
                                            scale_pool=scale_pool, layer_id=ii + 1).cuda()
            )
        self.layers.append(
            GradlessGCReplayNonlinBlock(out_channel=out_channel, in_channel=interm_channel, scale_pool=scale_pool,
                                        layer_id=n_layer - 1, use_act=False).cuda()
        )

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x_in):
        x_in = x_in.repeat(1, self.in_channel, 1, 1, 1)
        if isinstance(x_in, list):
            x_in = torch.cat(x_in, dim=0)

        nb, nc, nx, ny, nz = x_in.shape

        alphas = torch.rand(nb)[:, None, None, None]  # nb, 1, 1, 1

        x = self.layers[0](x_in)
        for blk in self.layers[1:]:
            x = blk(x)
        mixed = x

        mixed = imgnorm_torch(mixed)

        return mixed
