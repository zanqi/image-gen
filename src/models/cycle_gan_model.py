import torch
from collections import OrderedDict
from . import networks


class CycleGANModel():
    """
    CycleGAN model translate picture to picture. Trained using unpaired input:
    "This group of pictures should map to this other group of pictures".
    """

    def __init__(self, opt) -> None:
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
        self.direction = opt.direction
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and len(self.gpu_ids) > 0
            else 'cpu')

        self.net_g = networks.defineG(opt.input_nc, opt.output_nc, opt.ngf,
                                      self.gpu_ids)
        self.visual_names = []

    def set_input(self, ab_images):
        a_to_b = self.direction == 'AtoB'
        self.real_from = ab_images['A' if a_to_b else 'B'].to(self.device)
        self.real_to = ab_images['B' if a_to_b else 'A'].to(self.device)

    def optimize_parameters(self):
        self.forward()      # compute fake images and reconstruction images

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