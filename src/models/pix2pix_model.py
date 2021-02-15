import torch
from . import networks


class Pix2PixModel(object):
    """
    Pix2Pix model translate picture to picture. Trained using paired input: "This picture should map to this other picture".
    """

    def __init__(self, opt):
        object.__init__(self)
        self.isTrain = opt.isTrain
        self.direction = opt.direction
        self.netG = networks.defineG(opt.input_nc, opt.output_nc, opt.ngf)
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(
            self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        if self.isTrain:
            self.netD = networks.defineD(opt.input_nc + opt.output_nc, opt.ndf, 'basic')

            self.criterionGAN = networks.GANLoss(opt.gan_mode)
            self.critenrionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def set_input(self, input):
        AtoB = self.direction == 'AtoB'
        self.real_from = input['A' if AtoB else 'B'].to(self.device)
        self.real_to = input['B' if AtoB else 'A'].to(self.device)

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()
        self.backward_D()

    def forward(self):
        self.fake_to = self.netG(self.real_from)  # G(A)

    def set_requires_grad(self, net, require_grad):
        for param in net.parameters():
            param.requires_grad = require_grad

    def backward_D(self):
        fake_AB = torch.cat((self.real_from, self.fake_to), 1)
        pred_fake = self.netD(fake_AB.detach())
