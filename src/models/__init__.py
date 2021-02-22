"""
NN models
"""
import importlib


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


def set_requires_grad(net, require_grad):
    for param in net.parameters():
        param.requires_grad = require_grad
