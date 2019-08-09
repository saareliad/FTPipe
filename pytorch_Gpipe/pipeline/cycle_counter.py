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

    def current_input_valid(self, gpu_num: int):
        """
        Current input is considered valid in forward mode in cycles:
            [gpu_num, ..., gpu_num + num_runs]
        And in backward mode in cycles:
            [num_gpus - gpu_num - 1, ..., num_gpus - gpu_num - 1 + num_runs]

        So, for GPU #0 input will be valid for the first num_runs cycles.
        """
        if gpu_num is 0:
            return self.current_input_valid(1)

        if self.cur_mode is ForwardMode.backward:
            first_valid_iter = self.num_gpus - gpu_num - 1
        else:
            first_valid_iter = gpu_num - 1

        return first_valid_iter <= self.__counter < first_valid_iter + self.num_runs

    def prev_input_valid(self, gpu_num: int):
        """Checks if the input was valid one clock cycle ago"""
        if gpu_num is 0:
            return self.prev_input_valid(1)

        # it is equivalent to checking if the input is valid at the next GPU
        return self.current_input_valid(gpu_num + 1)
