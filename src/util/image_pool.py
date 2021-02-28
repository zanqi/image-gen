import random
import torch

class ImagePool():
    """
    A collection of images. Can get an random image out of it.
    """

    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)
            else:
                prob = random.uniform(0, 1)
                if prob > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return torch.stack(return_images)
