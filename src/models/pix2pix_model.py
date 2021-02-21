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
        self.is_train = opt.isTrain
        self.direction = opt.direction
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and len(self.gpu_ids) > 0
            else 'cpu')
        print(f"Device using: {self.device}")
        self.visual_names = ['real_from', 'fake_to', 'real_to']

        self.net_g = networks.defineG(
            opt.input_nc, opt.output_nc, opt.ngf, gpu_ids=self.gpu_ids)
        self.real_from = None
        self.real_to = None
        self.fake_to = None
        self.loss_d_fake = None
        self.loss_d_real = None
        self.loss_d = None
        self.loss_g_gan = None
        self.loss_g_l1 = None
        self.loss_g = None

        if self.is_train:
            self.net_d = networks.defineD(
                opt.input_nc + opt.output_nc, opt.ndf, 'basic',
                gpu_ids=self.gpu_ids)

            self.criterion_gan = GANLoss(opt.gan_mode).to(self.device)
            self.criterion_l1 = torch.nn.L1Loss()
            self.optimizer_g = torch.optim.Adam(
                self.net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_d = torch.optim.Adam(
                self.net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def set_input(self, x_input):
        a_to_b = self.direction == 'AtoB'
        self.real_from = x_input['A' if a_to_b else 'B'].to(self.device)
        self.real_to = x_input['B' if a_to_b else 'A'].to(self.device)

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.net_d, True)  # enable backprop for D
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()

        # D requires no gradients when optimizing G
        self.set_requires_grad(self.net_d, False)
        self.optimizer_g.zero_grad()        # set G's gradients to zero
        self.backward_g()                   # calculate gradients for G
        self.optimizer_g.step()             # udpate G's weights

    def forward(self):
        self.fake_to = self.net_g(self.real_from)  # G(A)

    def set_requires_grad(self, net, require_grad):
        for param in net.parameters():
            param.requires_grad = require_grad

    def backward_d(self):
        fake_ab = torch.cat((self.real_from, self.fake_to), 1)
        pred_fake = self.net_d(fake_ab.detach())
        self.loss_d_fake = self.criterion_gan(pred_fake, False)

        real_ab = torch.cat((self.real_from, self.real_to), 1)
        pred_real = self.net_d(real_ab)
        self.loss_d_real = self.criterion_gan(pred_real, True)

        self.loss_d = (self.loss_d_fake + self.loss_d_real) * 0.5
        self.loss_d.backward()

    def backward_g(self):
        fake_ab = torch.cat((self.real_from, self.fake_to), 1)
        pred_fake = self.net_d(fake_ab)
        self.loss_g_gan = self.criterion_gan(pred_fake, True)

        self.loss_g_l1 = self.criterion_l1(self.fake_to, self.real_to) * 100
        self.loss_g = self.loss_g_gan + self.loss_g_l1
        self.loss_g.backward()

    def get_current_visuals(self):
        visuals = OrderedDict()
        for name in self.visual_names:
            visuals[name] = getattr(self, name)
        return visuals
