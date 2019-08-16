from typing import Union

from .forward_mode import ForwardMode


class CycleCounter:
    """
    This class is responsible for counting the cycles of the current run of the
    pipelined data. It also used to decide whether a GPU should be working at
    each cycle.

    :param num_gpus: total amount of GPUs in the pipelined module.
    :param cur_mode: the initial mode (train/production or backward).
        default: train.
    """

    def __init__(self, num_gpus: int, cur_mode: ForwardMode = ForwardMode.train):
        self.__counter = 0
        self.cur_mode = cur_mode
        self.num_gpus = num_gpus
        self.num_runs = 0

    def set_num_runs(self, num_runs: int):
        self.num_runs = num_runs

    def change_mode(self, mode: Union[str, ForwardMode]):
        """
        Changes the mode of the forward propagation and resets the counter.
        """
        assert isinstance(mode, (str, ForwardMode))

        if isinstance(mode, str):
            self.cur_mode = ForwardMode[mode]
        else:
            self.cur_mode = mode

        self.reset()

    def reset(self):
        """resets the counter to zero"""

        self.__counter = 0
        self.num_runs = 0

    def tick(self):
        self.__counter += 1

    def get_count(self):
        return self.__counter

    def input_valid(self, gpu_num, cycle=0):
        """
        Checks if the input of the GPU is valid.

        :param gpu_num: the index of the wanted GPU
        :param cycle: specify at which cycle (relative to the current) to check
        """
        if gpu_num is 0:
            return self.input_valid(1, cycle)

        if self.cur_mode is ForwardMode.backward:
            first_valid_iter = self.num_gpus - gpu_num
        else:
            first_valid_iter = gpu_num - 1

        return first_valid_iter <= self.__counter + cycle < first_valid_iter + self.num_runs

    def output_valid(self, gpu_num, cycle=0):
        """
        Checks if the GPU should output a valid output.

        :param gpu_num: the index of the wanted GPU
        :param cycle: specify at which cycle (relative to the current) to check
        """
        if self.cur_mode is ForwardMode.backward:
            first_valid_iter = self.num_gpus - gpu_num - 1
        else:
            first_valid_iter = gpu_num

        return first_valid_iter <= self.__counter + cycle < first_valid_iter + self.num_runs
