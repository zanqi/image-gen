import torch.nn as nn

class UnetSkipConnectionBlock(nn.Module):
    """
    docstring
    """

    def __init__(self, outer_nc, inner_nc, input_nc, submodule, is_outermost, is_innermost, norm_layer, use_dropout):
        """
        docstring
        """
        pass