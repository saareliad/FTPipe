
from models import AlexNet
from modelParallel import PipeLineParallel


def test_structure():
    net = AlexNet()

    pipe_net = PipeLineParallel(net, DEBUG=True)

    print(pipe_net)
