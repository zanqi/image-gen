from PIL import Image
import os
import numpy as np


def tensor2im(image):
    image_numpy = image[0].detach().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(np.uint8)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_image(image_numpy, path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(path)
