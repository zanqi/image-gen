import torch
from src.data.aligned_dataset import AlignedDataset


def create_dataloader():
    data_set = AlignedDataset()
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=10,
        shuffle=False,
        num_workers=1)
    return data_loader
