import importlib

import torch
import torch.nn as nn
from torch.nn import init

import matplotlib.pyplot as plt
import random

from unet3d.buildingblocks import conv3d, Encoder, Decoder, FinalConv, DoubleConv, \
    ExtResNetBlock, SingleConv, GreenBlock, DownBlock, UpBlock, VaeBlock
from unet3d.utils import create_feature_maps


class UNet3D(nn.Module):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8,
                 **kwargs):
        super(UNet3D, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x)

        return x


class ResidualUNet3D(nn.Module):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock instead of DoubleConv as a basic building block as well as summation joining instead
    of concatenation joining. Since the model effectively becomes a residual net, in theory it allows for deeper UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. NLLLoss (multi-class)
            or BCELoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4,5
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        conv_layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=32, conv_layer_order='cge', num_groups=8,
                 **kwargs):
        super(ResidualUNet3D, self).__init__()

        if isinstance(f_maps, int):
            # use 5 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=5)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses ExtResNetBlock as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=ExtResNetBlock,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=ExtResNetBlock,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses ExtResNetBlock as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            decoder = Decoder(reversed_f_maps[i], reversed_f_maps[i + 1], basic_module=ExtResNetBlock,
                              conv_layer_order=conv_layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x)

        return x


class Noise2NoiseUNet3D(nn.Module):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock instead of DoubleConv as a basic building block as well as summation joining instead
    of concatenation joining. Since the model effectively becomes a residual net, in theory it allows for deeper UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. NLLLoss (multi-class)
            or BCELoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4,5
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, f_maps=16, num_groups=8, **kwargs):
        super(Noise2NoiseUNet3D, self).__init__()

        # Use LeakyReLU activation everywhere except the last layer
        conv_layer_order = 'clg'

        if isinstance(f_maps, int):
            # use 5 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=5)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=conv_layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # 1x1x1 conv + simple ReLU in the final convolution
        self.final_conv = SingleConv(f_maps[0], out_channels, kernel_size=1, order='cr', padding=0)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        return x


def get_model(config):
    def _model_class(class_name):
        m = importlib.import_module('unet3d.model')
        clazz = getattr(m, class_name)
        return clazz

    assert 'model' in config, 'Could not find model configuration'
    model_config = config['model']
    model_class = _model_class(model_config['name'])
    return model_class(**model_config)


###############################################Supervised Tags 3DUnet###################################################

class TagsUNet3D(nn.Module):
    """
    Supervised tags 3DUnet
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels; since most often we're trying to learn
            3D unit vectors we use 3 as a default value
        output_heads (int): number of output heads from the network, each head corresponds to different
            semantic tag/direction to be learned
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `DoubleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
    """

    def __init__(self, in_channels, out_channels=3, output_heads=1, conv_layer_order='crg', init_channel_number=32,
                 **kwargs):
        super(TagsUNet3D, self).__init__()

        # number of groups for the GroupNorm
        num_groups = min(init_channel_number // 2, 32)

        # encoder path consist of 4 subsequent Encoder modules
        # the number of features maps is the same as in the paper
        self.encoders = nn.ModuleList([
            Encoder(in_channels, init_channel_number, apply_pooling=False, conv_layer_order=conv_layer_order,
                    num_groups=num_groups),
            Encoder(init_channel_number, 2 * init_channel_number, conv_layer_order=conv_layer_order,
                    num_groups=num_groups),
            Encoder(2 * init_channel_number, 4 * init_channel_number, conv_layer_order=conv_layer_order,
                    num_groups=num_groups),
            Encoder(4 * init_channel_number, 8 * init_channel_number, conv_layer_order=conv_layer_order,
                    num_groups=num_groups)
        ])

        self.decoders = nn.ModuleList([
            Decoder(4 * init_channel_number + 8 * init_channel_number, 4 * init_channel_number,
                    conv_layer_order=conv_layer_order, num_groups=num_groups),
            Decoder(2 * init_channel_number + 4 * init_channel_number, 2 * init_channel_number,
                    conv_layer_order=conv_layer_order, num_groups=num_groups),
            Decoder(init_channel_number + 2 * init_channel_number, init_channel_number,
                    conv_layer_order=conv_layer_order, num_groups=num_groups)
        ])

        self.final_heads = nn.ModuleList(
            [FinalConv(init_channel_number, out_channels, num_groups=num_groups) for _ in
             range(output_heads)])

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        # apply final layer per each output head
        tags = [final_head(x) for final_head in self.final_heads]

        # normalize directions with L2 norm
        return [tag / torch.norm(tag, p=2, dim=1).detach().clamp(min=1e-8) for tag in tags]


################################################Distance transform 3DUNet##############################################
class DistanceTransformUNet3D(nn.Module):
    """
    Predict Distance Transform to the boundary signal based on the output from the Tags3DUnet. Fore training use either:
        1. PixelWiseCrossEntropyLoss if the distance transform is quantized (classification)
        2. MSELoss if the distance transform is continuous (regression)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. NLLLoss (multi-class)
            or BCELoss (two-class) respectively)
        final_sigmoid (bool): 'sigmoid'/'softmax' whether element-wise nn.Sigmoid or nn.Softmax should be applied after
            the final 1x1 convolution
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, init_channel_number=32, **kwargs):
        super(DistanceTransformUNet3D, self).__init__()

        # number of groups for the GroupNorm
        num_groups = min(init_channel_number // 2, 32)

        # encoder path consist of 4 subsequent Encoder modules
        # the number of features maps is the same as in the paper
        self.encoders = nn.ModuleList([
            Encoder(in_channels, init_channel_number, apply_pooling=False, conv_layer_order='crg',
                    num_groups=num_groups),
            Encoder(init_channel_number, 2 * init_channel_number, pool_type='avg', conv_layer_order='crg',
                    num_groups=num_groups)
        ])

        self.decoders = nn.ModuleList([
            Decoder(3 * init_channel_number, init_channel_number, conv_layer_order='crg', num_groups=num_groups)
        ])

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(init_channel_number, out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, inputs):
        # allow multiple heads
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            x = torch.cat(inputs, dim=1)
        else:
            x = inputs

        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        # apply final 1x1 convolution
        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x)

        return x


class EndToEndDTUNet3D(nn.Module):
    def __init__(self, tags_in_channels, tags_out_channels, tags_output_heads, tags_init_channel_number,
                 dt_in_channels, dt_out_channels, dt_final_sigmoid, dt_init_channel_number,
                 tags_net_path=None, dt_net_path=None, **kwargs):
        super(EndToEndDTUNet3D, self).__init__()

        self.tags_net = TagsUNet3D(tags_in_channels, tags_out_channels, tags_output_heads,
                                   init_channel_number=tags_init_channel_number)
        if tags_net_path is not None:
            # load pre-trained TagsUNet3D
            self.tags_net = self._load_net(tags_net_path, self.tags_net)

        self.dt_net = DistanceTransformUNet3D(dt_in_channels, dt_out_channels, dt_final_sigmoid,
                                              init_channel_number=dt_init_channel_number)
        if dt_net_path is not None:
            # load pre-trained DistanceTransformUNet3D
            self.dt_net = self._load_net(dt_net_path, self.dt_net)

    @staticmethod
    def _load_net(checkpoint_path, model):
        state = torch.load(checkpoint_path)
        model.load_state_dict(state['model_state_dict'])
        return model

    def forward(self, x):
        x = self.tags_net(x)
        return self.dt_net(x)


class TestTheNet(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8,
                 **kwargs):
        super(TestTheNet, self).__init__()

        self.greenblock = GreenBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.greenblock(x)
        return x


class VaeUNet(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='cgr', num_groups=8,
                 **kwargs):
        super(VaeUNet, self).__init__()
        self.conv3d = conv3d(in_channels, 32, kernel_size=3, bias=True)
        self.dropout = nn.Dropout(p=0.2)
        self.convblock = SingleConv(32, 32)
        self.downblock1 = DownBlock(32, 16)
        self.downblock2 = DownBlock(16, 32)
        self.downblock3 = DownBlock(32, 64)
        self.greenblock1 = GreenBlock(64, 64)
        self.greenblock2 = GreenBlock(64, 64)
        self.upblock1 = UpBlock(64, 32)
        self.convblock1 = SingleConv(96, 32, order=layer_order, num_groups=num_groups)
        self.convblock2 = SingleConv(32, 32, order=layer_order, num_groups=num_groups)
        self.upblock2 = UpBlock(32, 16)
        self.convblock3 = SingleConv(48, 16, order=layer_order, num_groups=num_groups)
        self.convblock4 = SingleConv(16, 16, order=layer_order, num_groups=num_groups)
        self.upblock3 = UpBlock(16, 8)
        self.convblock5 = SingleConv(24, 4, order=layer_order, num_groups=num_groups)
        self.convblock6 = SingleConv(4, 4, order=layer_order, num_groups=num_groups)
        self.final_conv = conv3d(4, 4, kernel_size=1, bias=True, padding=0)

        self.vae_block = VaeBlock(64, 4)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv3d(x)
        x = self.dropout(x)
        x = self.convblock(x)

        level1, l1_conv = self.downblock1(x)
        level2, l2_conv = self.downblock2(level1)
        level3, l3_conv = self.downblock3(level2)

        conv1 = self.greenblock1(level3)
        conv2 = self.greenblock2(conv1)

        level3_up = self.upblock1(conv2)
        concat = torch.cat([level3_up, l3_conv], 1)
        level3_up = self.convblock1(concat)
        level3_up = self.convblock2(level3_up)

        level2_up = self.upblock2(level3_up)
        concat = torch.cat([level2_up, l2_conv], 1)
        level2_up = self.convblock3(concat)
        level2_up = self.convblock4(level2_up)

        level1_up = self.upblock3(level2_up)
        concat = torch.cat([level1_up, l1_conv], 1)
        level1_up = self.convblock5(concat)
        level1_up = self.convblock6(level1_up)

        output = self.final_conv(level1_up)
        vae_out, z_mean, z_var = self.vae_block(conv2)

        if not self.training:
            output = self.final_activation(output)
        return output, vae_out, z_mean, z_var



# initalize the module
def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

# compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

class unetConv3(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv3, self).__init__()
        self.n = n

        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, stride, padding),
                                     nn.BatchNorm3d(out_size),
                                     nn.ReLU(inplace=True))
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size
        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, stride, padding),
                                     nn.ReLU(inplace=True))
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        for m in self.children():
            init_weights(m, 'kaiming')

    def forward(self, input):
        x = input
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x

class unetUp(nn.Module):

    def __init__(self, in_size, out_size, is_deconv=False, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv3(in_size+(n_concat-2)*out_size, out_size, is_batchnorm=False)
        if is_deconv:
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                nn.Conv3d(in_size, out_size, kernel_size=1)
            )

        for m in self.children():
            if m.__class__.__name__.find('unetConv3') != -1:continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):

        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)


class UNet_Nested(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='cgr', num_groups=8,
                n_classes=4, feature_scale=3, is_deconv=False, is_batchnorm=True, is_ds=True, ** kwargs):

        super(UNet_Nested, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds                    # deep supervision

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        filters = [16, 32, 64, 128, 256]

        # downsampling
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        self.conv00 = unetConv3(self.in_channels, filters[0], self.is_batchnorm)
        self.conv10 = unetConv3(filters[0], filters[1], self.is_batchnorm)
        self.conv20 = unetConv3(filters[1], filters[2], self.is_batchnorm)
        self.conv30 = unetConv3(filters[2], filters[3], self.is_batchnorm)
        self.conv40 = unetConv3(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_1 = nn.Conv3d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv3d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv3d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv3d(filters[0], n_classes, 1)
        self.sigmoid = nn.Sigmoid()

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        x00 = self.conv00(inputs)
        maxpool0 = self.maxpool(x00)
        x10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool(x10)
        x20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool(x20)
        x30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool(x30)
        x40 = self.conv40(maxpool3)
        # column : 1
        x01 = self.up_concat01(x10, x00)
        x11 = self.up_concat11(x20, x10)
        x21 = self.up_concat21(x30, x20)
        x31 = self.up_concat31(x40, x30)
        # column : 2
        x02 = self.up_concat02(x11, x00, x01)
        x12 = self.up_concat12(x21, x10, x11)
        x22 = self.up_concat22(x31, x20, x21)
        # column : 3
        x03 = self.up_concat03(x12, x00, x01, x02)
        x13 = self.up_concat13(x22, x10, x11, x12)
        # column : 4
        x04 = self.up_concat04(x13, x00, x01, x02, x03)

        # output_sample = x04[0, 1, :, :, 80].cpu().detach().numpy()
        # fig = plt.figure()
        # feature_image = fig.add_subplot(1, 1, 1)
        # plt.imshow(output_sample, cmap="gray")
        # feature_image.set_title('output')
        # plt.savefig('/home/liujing/pytorch-3dunet/picture/{}.png'.format(str(random.randint(1, 1000))))
        # plt.close()

        # final layer
        final_1 = self.final_1(x01)
        final_2 = self.final_2(x02)
        final_3 = self.final_3(x03)
        final_4 = self.final_4(x04)

        final = (final_1 + final_2 + final_3 + final_4) / 4

        if self.is_ds:
            return self.sigmoid(final)
        else:
            return final_4
