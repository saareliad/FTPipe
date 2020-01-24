class BigBatchManager:
    def __init__(self, step_every, optimizer, base_lr_batch_size, bs_train):
        """
        Handles Linear learning rate scaling according to base_lr_batch_size.

        """
        self.step_every = step_every

        self.num_bwds = 0
        self.total_bwd_size = 0

        self.base_lr_batch_size = base_lr_batch_size
        self.optimizer = optimizer
        # self.factor_method = "From Steps"

    def update_on_bwd(self, batch_size=None):
        """ Should be called after each backward"""
        # TODO: make it easy to use the real batch size.
        self.num_bwds += 1

        # HACK:
        if batch_size is None:
            batch_size = self.bs_train

        self.total_bwd_size += batch_size

    def should_step(self):
        return self.num_bwds == self.step_every

    def mult_lr_b4_step(self):
        """ Should be called before step() """
        # Calaculate factor
        factor = self.total_bwd_size / self.base_lr_batch_size

        # reset state
        self.num_bwds = 0
        self.total_bwd_size = 0

        self.old_lrs = [g['lr'] for g in self.optimizer.param_groups]
        if factor > 0:
            for g in self.optimizer.param_groups:
                g['lr'] *= factor

    def return_to_base_lrs(self):
        """ Should be called after step() """
        for g, old_lr in zip(self.optimizer.param_groups, self.old_lrs):
            g['lr'] = old_lr
    
