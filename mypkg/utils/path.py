"""Module with generic useful functions such as to return main dir path."""

import os

from pathlib import Path


def get_file_path(filename):
    """Find filename in the relative directory `../data/` .

    Args:
        filename (str): file we're looking for in the ./data/ directory.

    Returns:
        str: absolute path to file "filename" in ./data/ dir.

    """
    root_dir = Path(__file__).parent.parent
    file_dir = os.path.join(str(root_dir), "data", filename)

    return file_dir


def get_data_path():
    """Return absolute path to `/data/`."""
    root_path = Path(__file__).parent.parent
    data_path = os.path.join(str(root_path), "data")
    return data_path

def get_absolute_path(relative_path="."):
    """Return absolute path given `relative_path`.

    Args:
        relative_path (str): path relative to 'here'.

    Returns:
        str: absolute path

    """
    here_dir = os.path.dirname(os.path.realpath("__file__"))
    abs_path = os.path.join(str(here_dir), relative_path)

    return abs_path


def get_sibling_path(folder):
    """returns the path of 'folder' on the same level"""
    root_dir = Path(__file__).parent.parent
    sibling_dir = os.path.join(str(root_dir), folder)
    return sibling_dir


def get_root_path():
    root_path = Path(__file__).parent.parent
    return root_path

