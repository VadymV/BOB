"""
Miscellaneous functions.
"""

import pickle


def load_pickle(path_to_file: str) -> object:
    """
    Loads a serialized representation of a Python object.
    :param path_to_file: Path to a file.
    :return: Loaded object.
    """
    with open(path_to_file, "rb") as f:
        return pickle.load(f)
