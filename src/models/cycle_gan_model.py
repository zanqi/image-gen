from collections import OrderedDict
import itertools
import torch

from src.util.image_pool import ImagePool

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
        self.loss_d = None
        self.loss_d_reverse = None
        self.direction = opt.direction
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and len(self.gpu_ids) > 0
            else 'cpu')
        print(f"Device using: {self.device}")
        self.net_g = networks.define_g(opt.input_nc, opt.output_nc, opt.ngf, 'resnet',
                                       gpu_ids=self.gpu_ids)

        self.net_g_reverse = networks.define_g(opt.output_nc, opt.input_nc, opt.ngf, 'resnet',
                                               gpu_ids=self.gpu_ids)
        if self.is_train:
            self.net_d = networks.define_d(
                opt.output_nc, opt.ndf,
                gpu_ids=self.gpu_ids)

            self.net_d_reverse = networks.define_d(
                opt.input_nc, opt.ndf,
                gpu_ids=self.gpu_ids)

            self.criterion_gan = GANLoss(opt.gan_loss_mode).to(self.device)
            self.criterion_cycle = torch.nn.L1Loss()
            self.criterion_idt = torch.nn.L1Loss()

            self.optimizer_g = torch.optim.Adam(
                itertools.chain(
                    self.net_g.parameters(),
                    self.net_g_reverse.parameters()),
                lr=opt.learning_rate, betas=(opt.beta1, 0.999))
            self.optimizer_d = torch.optim.Adam(
                itertools.chain(
                    self.net_d.parameters(),
                    self.net_d_reverse.parameters()),
                lr=opt.learning_rate, betas=(opt.beta1, 0.999))
            self.fake_from_pool = ImagePool(opt.pool_size)
            self.fake_to_pool = ImagePool(opt.pool_size)
        self.visual_names = \
            ['real_from', 'fake_to', 'rec_from'] + \
            ['real_to', 'fake_from', 'rec_to']

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

        # Optimize D
        set_requires_grad(self.net_d, True)
        set_requires_grad(self.net_d_reverse, True)
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()

    def backward_d(self):
        fake_to = self.fake_to_pool.query(self.fake_to)
        self.loss_d = self.backward_d_base(
            self.net_d, self.real_to, fake_to)

        fake_from = self.fake_from_pool.query(self.fake_from)
        self.loss_d_reverse = self.backward_d_base(
            self.net_d_reverse, self.real_from, fake_from)

    def backward_d_base(self, net_d, real, fake):
        pred_real = net_d(real)
        loss_d_real = self.criterion_gan(pred_real, True)

        pred_fake = net_d(fake.detach())
        loss_d_fake = self.criterion_gan(pred_fake, False)

        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        return loss_d

    def backward_g(self):
        lambda_idt = self.opt.lambda_identity
        lambda_cycle = self.opt.lambda_cycle
        if lambda_idt > 0:
            idt_a = self.net_g(self.real_to)
            loss_idt_a = self.criterion_idt(idt_a, self.real_to) * lambda_cycle * lambda_idt
            idt_b = self.net_g_reverse(self.real_from)
            loss_idt_b = self.criterion_idt(idt_b, self.real_from) * lambda_cycle * lambda_idt
        else:
            loss_idt_a = 0
            loss_idt_b = 0
        loss_g_forward = self.criterion_gan(self.net_d(self.fake_to), True)
        self.loss_g_reverse = self.criterion_gan(
            self.net_d_reverse(self.fake_from), True)

        self.loss_cycle = self.criterion_cycle(
            self.rec_from, self.real_from) * lambda_cycle
        self.loss_cycle_reverse = self.criterion_cycle(
            self.rec_to, self.real_to) * lambda_cycle
        self.loss_g = loss_g_forward + self.loss_g_reverse + \
            self.loss_cycle + self.loss_cycle_reverse + loss_idt_a + loss_idt_b
        self.loss_g.backward()

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
