"""Helpers functions for python projects

Author: Magda Gregorova
Created: 2024-06-22
"""

import argparse
import re


class FileParser(argparse.ArgumentParser):
    """Tweaking original ArgumentParser to read key value pairs from lines.
    Ignores lines containing text in double brackets [xxx] and starting with hash #.
    """

    def convert_arg_line_to_args(self, arg_line):
        if re.match('\[.*\]', arg_line) or re.match('#.*', arg_line):
            return ''
        return arg_line.split()