# Utils
import os


def get_world_size() -> int:
    """Returns world size (from env), or 1 if not set"""
    return int(os.environ.get('WORLD_SIZE', 1))


def get_global_rank() -> int:
    """Returns global rank (from env), or 0 if not set"""
    return int(os.environ.get('RANK', 0))
