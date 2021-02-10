from src.data.image_folder import make_dataset
import torch.utils.data as data

class AlignedDataset():
    def __init__(self):
        dir = './datasets/facades/train'
        self.AB_paths = sorted(make_dataset(dir))
        return None
