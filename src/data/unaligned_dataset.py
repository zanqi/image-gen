import os
import torch.utils.data as data
from PIL import Image
from src.data import get_transform
from src.data.image_folder import make_dataset


class UnalignedDataset(data.Dataset):
    """
    Dataset that returns a pair of images from group A and B.
    """

    def __init__(self, opt):
        self.dir_a = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_b = os.path.join(opt.dataroot, opt.phase + 'B')

        self.a_paths = sorted(make_dataset(self.dir_a))
        self.b_paths = sorted(make_dataset(self.dir_b))
        self.a_size = len(self.a_paths)
        self.b_size = len(self.b_paths)
        assert self.a_size != 0, 'Input folder A is empty'
        assert self.b_size != 0, 'Input folder B is empty'

        self.transform = get_transform()

    def __len__(self):
        return max(self.a_size, self.b_size)

    def __getitem__(self, index):
        a_path = self.a_paths[index % self.a_size]  # make sure index is within then range
        b_path = self.b_paths[index % self.b_size]

        a_img = Image.open(a_path).convert('RGB')
        b_img = Image.open(b_path).convert('RGB')

        return {'A': self.transform(a_img), 'B': self.transform(b_img)}
