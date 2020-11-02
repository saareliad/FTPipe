from torchvision.datasets import CIFAR10, CIFAR100


if __name__ == "__main__":
    CIFAR100(root="", download=True, train=True)
    CIFAR100(root="", download=True, train=False)

    CIFAR10(root="", download=True, train=True)
    CIFAR10(root="", download=True, train=False)