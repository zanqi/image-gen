import importlib
import torch
from torchvision.transforms import transforms


def create_dataloader(opt):
    def find_dataset(dataset_mode):
        filename = "src.data." + dataset_mode + "_dataset"
        modellib = importlib.import_module(filename)
        model = None
        target_model_name = dataset_mode.replace('_', '') + 'Dataset'
        for name, cls in modellib.__dict__.items():
            if name.lower() == target_model_name.lower():
                model = cls
        assert model is not None, f"{target_model_name} is not found in source"
        return model
    data_set = find_dataset(opt.dataset_mode)
    data_set = data_set(opt)
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=10,
        shuffle=True,
        num_workers=4)
    return data_loader


def get_transform():
    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)
