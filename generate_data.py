"""Generating sport balls dataset 

Author: Magda Gregorova
Created: 2024-06-16
"""

import data.sportballs as sb
import argparse
import helpers


def get_config(cfile):
    parser = helpers.FileParser(fromfile_prefix_chars='@')
    # data
    parser.add_argument("--path", type=str, default="/home/magda/Data/SportBalls", help="Dir to store data.")
    parser.add_argument("--test", action="store_true", help="Flag for test data")
    parser.add_argument("-l", "--length", type=int, help="Length of dataset", default=10000)
    parser.add_argument("--c_size", type=int, help="Canvas size", default=64)
    parser.add_argument("--o_size", type=int, help="Object size", default=34)
    parser.add_argument("--n_objects", type=int, help="Number of objects", default=3)
    parser.add_argument("--task4", action="store_true", help="Flag for task 4")
    # args = parser.parse_args(['@./config.ini', '@'+cfile])
    args = parser.parse_args(['@'+cfile])
    args.train = False if args.test else True
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfile", type=str, default="config.ini", help="Config file for the particular run.")
    args = parser.parse_args()
    config = get_config(args.cfile)
    sb.store_sports_balls(
        length=config.length,
        train=config.train, 
        path=config.path,
        canvas_size = config.c_size,
        object_size = config.o_size,
        n_objects = config.n_objects,
        random_cls = config.task4
        )
