from collections import OrderedDict
import itertools
import torch

from .networks.gan_loss import GANLoss
from . import networks, set_requires_grad


class CycleGANModel():
    """
    CycleGAN model translate picture to picture. Trained using unpaired input:
    "This group of pictures should map to this other group of pictures".
    """

    def __init__(self, opt) -> None:
        self.opt = opt
        self.is_train = opt.is_train
        if self.is_train:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']
        self.real_from = None
        self.fake_to = None
        self.rec_from = None
        self.real_to = None
        self.fake_from = None
        self.rec_to = None
        self.loss_g = None
        self.loss_g_reverse = None
        self.loss_cycle = None
        self.loss_cycle_reverse = None
        self.direction = opt.direction
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and len(self.gpu_ids) > 0
            else 'cpu')

        self.net_g = networks.defineG(opt.input_nc, opt.output_nc, opt.ngf,
                                      self.gpu_ids)

        self.net_g_reverse = networks.defineG(opt.output_nc, opt.input_nc, opt.ngf,
                                              self.gpu_ids)
        if self.is_train:
            self.net_d = networks.defineD(
                opt.output_nc, opt.ndf, 'basic',
                gpu_ids=self.gpu_ids)

            self.net_d_reverse = networks.defineD(
                opt.input_nc, opt.ndf, 'basic',
                gpu_ids=self.gpu_ids)

            self.criterion_gan = GANLoss().to(self.device)
            self.criterion_cycle = torch.nn.L1Loss()

            self.optimizer_g = torch.optim.Adam(
                itertools.chain(self.net_g.parameters(),
                                self.net_g_reverse.parameters()),
                lr=opt.learning_rate, betas=(opt.beta1, 0.999))
        self.visual_names = []

    def set_input(self, ab_images):
        a_to_b = self.direction == 'AtoB'
        self.real_from = ab_images['A' if a_to_b else 'B'].to(self.device)
        self.real_to = ab_images['B' if a_to_b else 'A'].to(self.device)

    def optimize_parameters(self):
        self.forward()      # compute fake images and reconstruction images

        # Optimize G
        # Ds require no gradients when optimizing Gs
        set_requires_grad(self.net_d, False)
        set_requires_grad(self.net_d_reverse, False)
        self.optimizer_g.zero_grad()
        self.backward_g()             # calculate gradients for G_A and G_B
        self.optimizer_g.step()       # update G_A and G_B's weights

    def backward_g(self):
        self.loss_g = self.criterion_gan(self.net_d(self.fake_to), True)
        self.loss_g_reverse = self.criterion_gan(
            self.net_d_reverse(self.fake_from), True)

        self.loss_cycle = self.criterion_cycle(
            self.rec_from, self.real_from) * self.opt.lambda_cycle
        self.loss_cycle_reverse = self.criterion_cycle(
            self.rec_to, self.real_to) * self.opt.lambda_cycle

    def forward(self):
        self.fake_to = self.net_g(self.real_from)
        self.rec_from = self.net_g_reverse(self.fake_to)
        self.fake_from = self.net_g_reverse(self.real_to)
        self.rec_to = self.net_g(self.fake_from)

    def get_current_visuals(self):
        visuals = OrderedDict()
        for name in self.visual_names:
            visuals[name] = getattr(self, name)
        return visuals
