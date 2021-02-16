from src.data.image_folder import make_dataset
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms


class AlignedDataset(data.Dataset):
    def __init__(self):
        dir = './datasets/facades/train'
        self.AB_paths = sorted(make_dataset(dir))

    def __len__(self):
        return len(self.AB_paths)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        transform = self.get_transform()
        return {'A': transform(A), 'B': transform(B)}

    @staticmethod
    def get_transform():
        transform_list = []
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)
