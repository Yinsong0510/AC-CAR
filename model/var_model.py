import numpy as np
import torch
import torch.nn as nn

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


class UnetDecoder(nn.Module):
    """
    UNet decoder that takes encoder features from external source.

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
                 half_res=False):
        super().__init__()

        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        self.half_res = half_res

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

        # configure upsampling
        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # compute encoder output channels for skip connections
        encoder_nfs = [infeats]
        for level in range(self.nb_levels - 1):
            encoder_nfs.append(enc_nf[(level + 1) * nb_conv_per_level - 1])

        # bottleneck input is concatenation of two encoders
        prev_nf = encoder_nfs[-1] * 2
        encoder_nfs = np.flip(encoder_nfs)

        # build decoder layers
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

    def forward(self, x_list, y_list, x):
        # decoder forward pass with upsampling and skip connections
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_list.pop(), y_list.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class Variance_Decoder(LoadableModel):
    """Variance estimation network that outputs log-variance."""

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        super().__init__()

        self.training = True

        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # build decoder
        self.unet_model = UnetDecoder(
            inshape,
            infeats=src_feats + trg_feats,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # variance output layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.variance = Conv(self.unet_model.final_nf, 1, kernel_size=3, padding=1)

    def forward(self, x, x_list, y_list):
        x = self.unet_model(x_list, y_list, x)
        log_variance = self.variance(x)
        return log_variance
