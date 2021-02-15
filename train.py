from src.models import create_model
from src.data import create_dataloader


class Opt(object):
    def __init__(self, isTrain, gan_mode):
        self.isTrain = isTrain
        self.gan_mode = gan_mode
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.lr = 0.0002
        self.beta1 = 0.5
        self.ndf = 64
        self.direction = 'AtoB'
        self.gpu_ids = []


if __name__ == '__main__':
    opt = Opt(True, 'basic')
    net = create_model(opt)
    dataloader = create_dataloader()
    for epoch in range(1, 101):
        for i, data in enumerate(dataloader):
            net.set_input(data)
            net.optimize_parameters()
            print(f"epoch {epoch}, batch {i}")
