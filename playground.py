#!/usr/bin/python

import os
import util


# def get_path(filename):
#     """Return file's path or empty string if no path."""
#     head, tail = os.path.split(filename)
#     return head


filename = __file__
breakpoint()
filename_path = util.get_path(filename)
print(f'path = {filename_path}')