from collections import deque


def conveyor_gen(inputs, queue):
    for x in inputs:
        yield x

    while not all([x is None for x in queue]):
        yield None


def prod_line(inputs, actions, size, output_results=True, first_ac=lambda x: x, last_ac=lambda x: x):
    assert size == len(actions)

    outputs = deque([]) if output_results else None

    actions = [lambda x: x if x is None else act(x) for act in actions]

    first_ac = lambda x: x if x is None else first_ac(x)
    last_ac = lambda x: x if x is None else last_ac(x)

    belt = deque([None for _ in range(size - 1)])

    for x in conveyor_gen(inputs, belt):
        belt.appendleft(first_ac(x))

        for i, act in enumerate(actions):
            belt[i] = act(belt[i])

        last_val = last_ac(belt.pop())

        if output_results and last_val is not None:
            outputs.appendleft(last_val)

    return outputs
