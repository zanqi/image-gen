import torch.utils.data as data

from PIL import Image
import os

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), f"{dir} is not a valid dir"