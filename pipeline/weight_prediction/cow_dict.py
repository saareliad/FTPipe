# Taken from https://github.com/csernazs/cowdict/blob/master/cowdict/cowdict.py
from collections import MutableMapping


class CowDict(MutableMapping):
    def __init__(self, base: dict):
        self.base = base
        self.dict = {}
        self.deleted_keys = set()

    def __getitem__(self, key):
        if key in self.deleted_keys:
            raise KeyError(key)

        try:
            return self.dict[key]
        except KeyError:
            return self.base[key]

    def __setitem__(self, key, value):
        try:
            self.deleted_keys.remove(key)
        except KeyError:
            pass

        self.dict[key] = value

    def __delitem__(self, key):
        if key in self.base:
            try:
                del self.dict[key]
            except KeyError:
                pass

            self.deleted_keys.add(key)

        elif key in self.dict:
            del self.dict[key]
            self.deleted_keys.add(key)
        else:
            raise KeyError(key)

    def __len__(self):
        return len(set(self.dict.keys()).union(set(self.base.keys())) - self.deleted_keys)

    def __iter__(self):

        for key in self.dict:
            if key not in self.deleted_keys:
                yield key

        for key in self.base:
            if key not in self.dict and key not in self.deleted_keys:
                yield key

    def __repr__(self):
        retval = ["{"]
        for key, value in self.items():
            retval.append(repr(key))
            retval.append(": ")
            retval.append(repr(value))
            retval.append(", ")

        del retval[-1]
        retval.append("}")
        return "".join(retval)
