import multiprocessing as mp
from .stage import Stage


class PipeLine():
    def __init__(self):
        super(PipeLine, self).__init__()
        self.stages = []
        self.in_buffer = mp.Queue()
        self.out_buffer = mp.Queue()

        def retrieve_output(buffer):
            while True:
                yield buffer.get()

        self.out_worker = mp.Process(
            target=retrieve_output, args=(self.out_buffer,), daemon=True)

    def add_stage(self, func, filter_stage=False, num_workers=1):
        if self.stages == []:
            stage = Stage(func=func, filter_stage=filter_stage, in_buffer=self.in_buffer,
                          out_buffer=self.out_buffer, num_workers=num_workers)
        else:
            transition_queue = mp.Queue()
            self.stages[-1].out_buffer = transition_queue
            stage = Stage(func=func, filter_stage=filter_stage, in_buffer=transition_queue,
                          out_buffer=self.out_buffer, num_workers=num_workers)

        self.stages.append(stage)

    def start(self):
        for stage in self.stages:
            stage.start()

        self.out_worker.start()

    def stop(self):
        self.out_worker.close()
        for stage in self.stages:
            stage.stop()

    def get(self, block=True, timeout=None):
        return self.out_buffer.get(block=block, timeout=timeout)

    def put(self, item, block=True, timeout=None):
        self.in_buffer.put(item, block=block, timeout=timeout)

    def feed_pipe(self, input_sequence):
        def feed(seq):
            for sample in seq:
                self.put(sample)

        feed_worker = mp.Process(
            target=feed, args=(input_sequence,), daemon=True)

        feed_worker.start()
