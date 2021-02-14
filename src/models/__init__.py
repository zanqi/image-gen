
from .pix2pix_model import Pix2PixModel

def create_model(opt):
    """
    Create model object based on options
    """
    instance = Pix2PixModel(opt)
    return instance