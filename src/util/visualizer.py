import os
import src.util as util


class Visualizer():
    def __init__(self, opt) -> None:
        self.web_dir = os.path.join(opt.checkpoints_dir, opt.model, 'web')
        self.img_dir = os.path.join(self.web_dir, 'images')
        print('create web directory %s...' % self.web_dir)
        util.mkdirs([self.web_dir, self.img_dir])

    def display_current_results(self, visuals, epoch, save_result):
        for label, image in visuals.items():
            image_numpy = util.tensor2im(image)
            img_path = os.path.join(
                self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
            util.save_image(image_numpy, img_path)
