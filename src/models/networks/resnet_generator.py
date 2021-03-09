import torch.nn as nn

from src.models.networks.resnet_block import ResnetBlock


class ResnetGenerator(nn.Module):
    """
    Resnet-based generator
    """

    def __init__(self, input_nc, output_nc, ngf=64, num_resblock=3, norm_layer=nn.BatchNorm2d):
        super().__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                          stride=2, padding=1, bias=False),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)]
        mult = 2 ** n_downsampling

        for i in range(num_resblock):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult,
                                  norm=norm_layer)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)

            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=False),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input_x):
        return self.model(input_x)
