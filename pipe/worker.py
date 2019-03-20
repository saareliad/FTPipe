import multiprocessing as mp


class Worker(mp.Process):
    def __init__(self, in_queue, out_queue, func, filter_stage):
        super(Worker, self).__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.func = func
        self.filter_stage = filter_stage

    def run(self):
        while True:
            data = self.in_queue.get()
            if data != None:
                if self.filter_stage:
                    if self.func(data):
                        self.out_queue.put(data)
                else:
                    self.out_queue.put(self.func(data))
            else:
                self.in_queue.put(None)
                break
