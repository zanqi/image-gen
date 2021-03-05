import torch.nn as nn


class ResnetBlock(nn.Module):
    """
    Resnet Block
    """

    def __init__(self, dim, norm):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, norm)

    def build_conv_block(self, dim, norm):
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3,
                                 padding=0, bias=False), norm(dim), nn.ReLU(True)]
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3,
                                 padding=0, bias=False), norm(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x_input):
        out = x_input + self.conv_block(x_input)  # add skip connections
        return out
