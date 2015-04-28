#!/usr/bin/env python3
import sys, os
from settings import *
import argparse
import init
import importlib

class Main():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--log', type=int, action='store',
                            dest='log', default=1,
                            help='Log levels, [1] = debug, [2] = info, ...')
        subpars = parser.add_subparsers(dest='verb')

        init_parser = subpars.add_parser('init')
        run_parser = subpars.add_parser('run')

        run_parser.add_argument('profile', type=str, action='store',
                                help='Profile name')

        args = parser.parse_args()

        if args.log != 1:
            lg = logging.getLogger()
            lg.setLevel(args.log * 10)
        del args.log

        if args.verb is not None:
            func = getattr(self, args.verb)
            if func is not None:
                del args.verb
                func(**vars(args))
        else:
            parser.print_help()

    def init(self):
        init.init()

    def run(self, profile):
        P = importlib.import_module('profiles.{}'.format(profile))
        ex = P.Executor()
        ex.start()



def init_log_settings():
    logging.basicConfig(format='[%(asctime)s.%(msecs)d] %(module)s - %(levelname)s : %(message)s',
                        datefmt='%H:%M:%S',
                        level=LOG_LEVEL)


def main():
    init_log_settings()
    Main()


main()
