from src.models import create_model
from src.data import create_dataloader
from src.util.visualizer import Visualizer


class Opt(object):
    """
    Command line options
    """

    def __init__(self, isTrain):
        self.is_train = isTrain
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.learning_rate = 0.0002
        self.beta1 = 0.5
        self.ndf = 64
        self.direction = 'AtoB'
        self.gpu_ids = []
        self.checkpoints_dir = './my_checkpoints'
        self.model = "cycle_gan"  # 'pix2pix'
        self.lambda_cycle = 10
        self.lambda_identity = 0.5
        self.dataset_mode = 'unaligned'
        self.dataroot = './datasets/horse2zebra'
        self.phase = 'train'
        self.gan_loss_mode = 'lsgan'
        self.pool_size = 50
        self.batch_size = 1
        self.num_dataloader_threads = 4


if __name__ == '__main__':
    opt = Opt(True)
    net = create_model(opt)
    viz = Visualizer(opt)
    dataloader = create_dataloader(opt)
    for epoch in range(1, 101):
        for i, data in enumerate(dataloader):
            net.set_input(data)
            net.optimize_parameters()
            if i % 20 == 0:
                print(f"epoch {epoch}, batch {i}")

        if epoch % 5 == 0:
            viz.display_current_results(net.get_current_visuals(), epoch)
