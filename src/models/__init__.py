"""
NN models
"""
import importlib
from .pix2pix_model import Pix2PixModel


def create_model(opt):
    """
    Create model object based on options
    """
    model = find_model(opt.model)
    instance = model(opt)
    return instance

def find_model(name):
    """
    Find model by name
    """
    model_filename = "src.models." + name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = name.replace('_', '') + 'Model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls
    assert model is not None, f"{target_model_name} is not found in source"
    return model
