import torch
import torch.nn as nn
from cfg import ShuffleNet_cfg as cfg


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

num_out = cfg['num_out']

class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=num_out, inverted_residual=InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
                                    nn.Conv2d(input_channels, 64, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                  )
        # self.conv2 = nn.Sequential(
        #                             nn.Conv2d(64, 64, 3, 1, 1, bias=False),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU(inplace=True),
        #                           )
        self.conv11 = nn.Sequential(
                                    nn.Conv2d(64, output_channels, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(output_channels),
                                    nn.ReLU(inplace=True),
                                  ) 
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        #output_channels = self._stage_out_channels[-1]

        self.conv5 = nn.Sequential(
                                    nn.ConvTranspose2d(232, 324, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(324),
                                    nn.ReLU(inplace=True),
                                  )
        self.conv6 = nn.Sequential(
                                    nn.ConvTranspose2d(324, 256, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                  )
        self.conv7 = nn.Sequential(
                                    nn.Conv2d(256, 128, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                  ) 
        self.conv8 = nn.Sequential(
                                    nn.Conv2d(128, 68, 3, 1, 1, bias=False),
                                    #nn.BatchNorm2d(68),
                                    nn.ReLU(inplace=True),
                                  )

        #self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        #x = self.conv2(x)
        x = self.conv11(x)
        x = self.maxpool(x)
        #print(x.shape)
        x = self.stage2(x)
        #print(x.shape)
        x = self.stage3(x)
        #print(x.shape) #[16, 232, 24, 24]
        x = self.conv5(x)
        #print(x.shape) #[16, 464, 48, 48]
        x = self.conv6(x)
        #print(x.shape)
        x = self.conv7(x)
        #print(x.shape)
        x = self.conv8(x)
        #print(x.shape) #[B , 68, 96, 96]
        return x

    def forward(self, x):
        return self._forward_impl(x)


def _shufflenetv2(arch, pretrained, progress, *args, **kwargs):
    model = ShuffleNetV2(*args, **kwargs)
    # if pretrained:
    #     model_url = model_urls[arch]
    #     if model_url is None:
    #         raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
    #     else:
    #         state_dict = load_state_dict_from_url(model_url, progress=progress)
    #         model.load_state_dict(state_dict)

    return model


def shufflenet_v2_x1_0(pretrained=False, progress=False, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model =  _shufflenetv2('shufflenetv2_x1.0', pretrained, progress,
                         [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)

    return model