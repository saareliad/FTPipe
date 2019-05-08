from collections import deque
from typing import Iterable, Callable, List, Union, NoReturn


def conveyor_gen(inputs: Iterable, queue: Iterable):
    """
    yields all of the stuff in inputs and then yields None until queue only has None values in it
    :param inputs: iterable of inputs
    :param queue: iterable that should be full with None values when the generator should stop
    :return: values in inputs and then None values
    """
    for x in inputs:
        yield x

    while not all([x is None for x in queue]):
        yield None


def prod_line(inputs: Iterable, actions: List[Callable], output_results: bool = True, first_ac: Callable = lambda x: x,
              last_ac: Callable = lambda x: x) -> Union[List, NoReturn]:
    """
    runs a pipeline with the inputs and actions
    :param inputs: pipeline inputs
    :param actions: pipeline functions - each action is what will happen in a 'station',
    the actions will be run on the inputs as they are ordered in the inputted iterable
    :param output_results: boolean for whether or not we want to return the outputs in the end,
    used mostly to save time and memory when possible
    :param first_ac: action that will be called on every input before it is put into the pipeline
    :param last_ac: action that will be called on every output after it leaves the pipeline
    :return: list of outputs of the pipeline if output_results == True, None otherwise
    """

    # calculate pipeline size
    size = len(actions)

    # outputs queue, None if not needed
    outputs = deque([]) if output_results else None

    # make sure the actions will ignore None inputs
    actions = [lambda x: x if x is None else act(x) for act in actions]
    first_ac_wrap = lambda x: x if x is None else first_ac(x)
    last_ac_wrap = lambda x: x if x is None else last_ac(x)

    # the actual pipeline, will start with the first station empty and None values in the rest
    belt = deque([None for _ in range(size - 1)])

    # as long as there are inputs left in either 'inputs' or in the belt (and for each input)
    for x in conveyor_gen(inputs, belt):
        # add the next input to the first station
        belt.appendleft(first_ac_wrap(x))

        # call action in each station
        for i, act in enumerate(actions):
            belt[i] = act(belt[i])

        # calculate the output of the last station and move every value to it's next station
        last_val = last_ac_wrap(belt.pop())

        # save said output if needed
        if output_results and last_val is not None:
            outputs.append(last_val)

    return outputs

