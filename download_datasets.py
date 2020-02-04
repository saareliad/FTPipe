from torchvision.datasets import CIFAR10, CIFAR100

DATA_DIR = "/home_local/saareliad/data"


CIFAR100(root=DATA_DIR, download=True,
         train=True)
CIFAR100(root=DATA_DIR, download=True,
         train=False)


CIFAR10(root=DATA_DIR, download=True,
        train=True)
CIFAR10(root=DATA_DIR, download=True,
        train=False)
