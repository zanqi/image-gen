import torch.nn as nn
import torch

class GANLoss(nn.Module):
    """
    Loss function for GAN, log(D(y)) and log(1- D(G(x))) in one class
    """

    def __init__(self, gan_mode):
        super(GANLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            return torch.tensor(1.0).expand_as(prediction)
        else:
            return torch.tensor(0.0).expand_as(prediction)