class CycleGANModel():
    """
    docstring
    """
    def __init__(self, opt) -> None:
        self.is_train = opt.isTrain
        if self.is_train:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']
