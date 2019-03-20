from .worker import Worker


class Stage():
    def __init__(self, func, in_buffer, out_buffer, filter_stage=False, num_workers=1):
        super(Stage, self).__init__()
        self.in_buffer = in_buffer
        self.num_workers = num_workers
        self.out_buffer = out_buffer
        self.workers = [
            Worker(self.in_buffer, self.out_buffer, func=func, filter_stage=filter_stage)]

    def start(self):
        for worker in self.workers:
            worker.daemon = True
            worker.start()

    def stop(self):
        self.in_buffer.put(None)

        for worker in self.workers:
            worker.join()
