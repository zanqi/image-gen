import torch.nn as nn
import torch


class GANLoss(nn.Module):
    """
    Loss function for GAN, log(D(y)) and log(1- D(G(x))) in one class
    """

    def __init__(self, gan_mode):
        super(GANLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            return self.real_label.expand_as(prediction)
        else:
            return self.fake_label.expand_as(prediction)
