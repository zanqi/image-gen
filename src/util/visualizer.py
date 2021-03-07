import os
import matplotlib.pyplot as plt
import src.util as util


class Visualizer():
    """
    Display or save the output images
    """

    def __init__(self, opt) -> None:
        self.web_dir = os.path.join(opt.checkpoints_dir, opt.model, 'web')
        self.img_dir = os.path.join(self.web_dir, 'images')
        print('create web directory %s...' % self.web_dir)
        util.mkdirs([self.web_dir, self.img_dir])

    def display_current_results(self, visuals, epoch):
        fig = plt.figure(figsize=(16, 16))
        for i, (label, image) in enumerate(visuals.items()):
            image_numpy = util.tensor2im(image)
            # img_path = os.path.join(
            #     self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
            # util.save_image(image_numpy, img_path)

            axis = fig.add_subplot(2, 3, i+1)
            axis.set_title(label)
            plt.imshow(image_numpy, interpolation='nearest')

        plt.show()
