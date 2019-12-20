from itertools import zip_longest

__all__ = ["grouper"]
# Creating iteration tool for "Double Buffers"


def zip_discard_compr(*iterables, sentinel=object()):
    # https://stackoverflow.com/questions/38054593/zip-longest-without-fillvalue
    return [[entry for entry in iterable if entry is not sentinel]
            for iterable in zip_longest(*iterables, fillvalue=sentinel)]


def grouper(iterable, n):
    """Collect data into *non fixed-length* chunks or blocks
        (changed the one in itertools recepies)
    """
    # grouper('ABCDEFG', 3,) --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_discard_compr(*args)


# Fixed recved:
# [torch.cat(group) for group in grouper(x, num_chunks)]

# [torch.cat(group) for group in grouper(x, self.comm_handler.num_chunks)]
