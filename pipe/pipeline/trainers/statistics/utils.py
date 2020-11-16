class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        # self.record = []

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def get_avg(self):
        return self.sum / self.count


class AccuracyMeter(AverageMeter):
    def __init__(self):
        super().__init__()

    def update(self, val, n=1):
        """ just to supoort adding num correct instead of accuracy """
        self.sum += val
        self.count += n

    def get_avg(self):
        return (self.sum / self.count) * 100
