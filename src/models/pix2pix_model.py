import torch
from . import networks

class Pix2PixModel(object):
    """
    Pix2Pix model translate picture to picture. Trained using paired input: "This picture should map to this other picture".
    """

    def __init__(self, opt):
        object.__init__(self)
        self.isTrain = opt.isTrain
        self.netG = networks.defineG(opt.input_nc, opt.output_nc, opt.ngf)

        if self.isTrain:
            self.netD = networks.defineD()

            self.criterionGAN = networks.GANLoss(opt.gan_mode)
            self.critenrionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
