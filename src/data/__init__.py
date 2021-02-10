
from src.data.aligned_dataset import AlignedDataset


def create_dataset():
    data_loader = AlignedDataset()
    data = data_loader.load_data()
    return data