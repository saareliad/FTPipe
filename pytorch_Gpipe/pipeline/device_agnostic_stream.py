import torch


class DeviceAgnosticStream:
    """
    This class is a device agnostic implementation of torch.Stream.
    It behaves the same as torch.Stream if given device is a cuda device, and
    doesn't do anything if the device is 'cpu'.
    """
    def __init__(self, device: str = None):
        if device == 'cpu':
            self.stream = None
        else:
            self.stream = torch.cuda.Stream(device=device)

    def __enter__(self):
        if self.stream is not None:
            return self.stream.__enter__()
        else:
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream is not None:
            return self.stream.__exit__()