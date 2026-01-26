import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from utils.layers import SpatialTransformer
from utils.modelio import LoadableModel, store_config_args, default_unet_features

class ConvBlock(nn.Module):
    """Convolutional block with LeakyReLU activation."""

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.main(x))


class CIN(nn.Module):
    """Conditional Instance Normalization."""

    def __init__(self, ndims, in_dim, inshape):
        super().__init__()
        IN = getattr(nn, 'InstanceNorm%dd' % ndims)
        self.norm = IN(in_dim)
        self.style = nn.Linear(inshape, in_dim * 2)
        self.style.bias.data[:in_dim] = 0
        self.style.bias.data[in_dim:] = 0

    def forward(self, input, low_x):
        low_x = torch.flatten(low_x, 1)
        style = self.style(low_x).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        gamma, beta = style.chunk(2, dim=1)
        out = self.norm(input)
        out = (1. + gamma) * out + beta
        return out


class ACFM(nn.Module):
    """Adaptive Conditional Feature Modulation block."""

    def __init__(self, ndims, in_dim, out_dim, inshape):
        super().__init__()
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.ai1 = CIN(ndims, in_dim, inshape)
        self.conv1 = Conv(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.ai2 = CIN(ndims, in_dim, inshape)
        self.conv2 = Conv(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, low_x):
        out = F.leaky_relu(self.ai1(x, low_x), negative_slope=0.2)
        out = self.conv1(out)
        out = self.conv2(F.leaky_relu(self.ai2(out, low_x), negative_slope=0.2))
        out += x
        return out


class ProjHead(nn.Module):
    """Projection head for contrastive learning."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.dense = nn.Conv3d(in_dim, out_dim, 1, stride=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        vec = self.avgpool(x)
        vec = torch.flatten(vec, 1)
        vec = self.fc(vec)
        dense = self.dense(x)
        vec = F.normalize(vec, dim=1)
        dense = F.normalize(dense, dim=1)
        return vec, dense


class Unet(nn.Module):
    """
    UNet architecture with ACFM encoder and standard decoder.
    Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels.
    """

    def __init__(self,
                 inshape,
                 infeats,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False,
                 out_dim=32):
        super().__init__()

        ndims = len(inshape)
        pixel_num = int(inshape[0] * inshape[1])
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        self.half_res = half_res
        self.slice_idx = inshape[-1] // 2 if inshape[-1] % 2 == 0 else (inshape[-1] - 1) // 2

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically from integer specification
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # parse feature configuration
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        # configure pooling and upsampling
        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # projection head for contrastive learning
        self.proj_head = ProjHead(enc_nf[-1], out_dim)

        # build encoder (down-sampling path with ACFM)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ACFM(ndims, prev_nf, nf, int(pixel_num / (4 ** level))))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # bottleneck input is concatenation of two encoders
        prev_nf = prev_nf * 2
        encoder_nfs = np.flip(encoder_nfs)

        # build decoder (up-sampling path)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += 2 * encoder_nfs[level]

        # full resolution convolutions
        self.remaining = nn.ModuleList()
        for nf in final_convs:
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        self.final_nf = prev_nf

    def forward(self, source, target, registration):
        # encoder forward pass
        src_history = [source]
        trg_history = [target]
        src_l = source[:, :, :, :, self.slice_idx]
        trg_l = target[:, :, :, :, self.slice_idx]

        for level, convs in enumerate(self.encoder):
            for conv in convs:
                source = conv(source, src_l)
                target = conv(target, trg_l)
            src_history.append(source)
            trg_history.append(target)
            source = self.pooling[level](source)
            target = self.pooling[level](target)
            src_l = F.avg_pool2d(src_l, kernel_size=2, stride=2)
            trg_l = F.avg_pool2d(trg_l, kernel_size=2, stride=2)

        # projection for contrastive learning
        src_vec, src_dense = self.proj_head(source.clone())
        trg_vec, trg_dense = self.proj_head(target.clone())

        src_list = [src.clone() for src in src_history]
        trg_list = [trg.clone() for trg in trg_history]

        x = torch.cat([source, target], dim=1)
        lat_x = x.clone()

        if registration:
            # decoder forward pass with upsampling and skip connections
            for level, convs in enumerate(self.decoder):
                for conv in convs:
                    x = conv(x)
                if not self.half_res or level < (self.nb_levels - 2):
                    x = self.upsampling[level](x)
                    x = torch.cat([x, src_history.pop(), trg_history.pop()], dim=1)

            # remaining convs at full resolution
            for conv in self.remaining:
                x = conv(x)

        return x, src_vec, trg_vec, src_dense, trg_dense, src_list, trg_list, lat_x


class VxmDense(LoadableModel):
    """VoxelMorph network for unsupervised nonlinear registration."""
    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 unet_half_res=False,
                 out_dim=None):
        super().__init__()

        self.training = True

        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # build UNet
        self.unet_model = Unet(
            inshape,
            infeats=1,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
            out_dim=out_dim,
        )

        # flow output layer (initialized with small weights)
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # spatial transformer for warping
        self.transformer = SpatialTransformer(inshape)

    def forward(self, X_aug, X, Y_aug, Y, registration):
        """
        Args:
            X_aug: Augmented source image tensor.
            X: Source image tensor.
            Y_aug: Augmented target image tensor.
            Y: Target image tensor (unused, kept for interface compatibility).
            registration: If True, return warped image and flow; otherwise return features.
        """
        x, src_vec, trg_vec, src_dense, trg_dense, src_list, trg_list, lat_x = self.unet_model(X_aug, Y_aug, registration)

        if registration:
            flow_field = self.flow(x)
            y_source = self.transformer(X, flow_field)
            y_source_aug = self.transformer(X_aug, flow_field)
            return y_source, y_source_aug, flow_field, src_vec, trg_vec, src_dense, trg_dense
        else:
            return lat_x, src_list, trg_list, src_vec, trg_vec, src_dense, trg_dense
