import os


def is_valid_dir(parser, arg):
    """
    Check if argument is existing directory.
    """
    if not os.path.isdir(arg) and arg is not None:
        parser.error("The directory {0} does not exist!".format(arg))

    return arg


def is_valid_slice(parser, arg):
    """
    Check if argument is valid slice.
    """
    parts = [int(part) for part in arg.split(":")]
    if len(parts) != 2 or not isinstance(parts[0], int) or not isinstance(parts[1], int):
        parser.error("Slice parameter must be of format <int>:<int>")

    return parts
