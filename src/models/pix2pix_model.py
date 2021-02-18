from collections import OrderedDict
import torch
from . import networks
from .networks.GANLoss import GANLoss


class Pix2PixModel(object):
    """
    Pix2Pix model translate picture to picture. Trained using paired input:
    "This picture should map to this other picture".
    """

    def __init__(self, opt):
        object.__init__(self)
        self.isTrain = opt.isTrain
        self.direction = opt.direction
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and len(self.gpu_ids) > 0
            else 'cpu')
        print(f"Device using: {self.device}")
        self.visual_names = ['real_from', 'fake_to', 'real_to']

        self.netG = networks.defineG(
            opt.input_nc, opt.output_nc, opt.ngf, gpu_ids=self.gpu_ids)

        if self.isTrain:
            self.netD = networks.defineD(
                opt.input_nc + opt.output_nc, opt.ndf, 'basic',
                gpu_ids=self.gpu_ids)

            self.criterionGAN = GANLoss(opt.gan_mode)
            self.criterionL1 = torch.nn.L1Loss()
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
        self.optimizer_D.step()

        # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate gradients for G
        self.optimizer_G.step()             # udpate G's weights

    def forward(self):
        self.fake_to = self.netG(self.real_from)  # G(A)

    def set_requires_grad(self, net, require_grad):
        for param in net.parameters():
            param.requires_grad = require_grad

    def backward_D(self):
        fake_AB = torch.cat((self.real_from, self.fake_to), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_from, self.real_to), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = torch.cat((self.real_from, self.fake_to), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_to, self.real_to) * 100
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def get_current_visuals(self):
        visuals = OrderedDict()
        for name in self.visual_names:
            visuals[name] = getattr(self, name)
        return visuals
