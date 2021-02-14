import torch.nn as nn


class GANLoss(nn.Module):
    """
    Loss function for GAN
    """

    def __init__(self, gan_mode):
        super(GANLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()