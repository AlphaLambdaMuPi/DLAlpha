#!/usr/bin/env python3
import sys, os
from settings import *
import argparse
import init, init_fuel
import importlib

class Main():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--log', type=int, action='store',
                            dest='log', default=1,
                            help='Log levels, [1] = debug, [2] = info, ...')
        subpars = parser.add_subparsers(dest='verb')

        init_parser = subpars.add_parser('init')
        init_fuel_parser = subpars.add_parser('init_fuel')
        run_parser = subpars.add_parser('run')

        run_parser.add_argument('profile', type=str, action='store',
                                help='Profile name')

        init_fuel_parser.add_argument('-v', '--validate-percentage', type=float, action='store',
                                help='percentage of validate datas', default=0.1, dest='valp')
        init_fuel_parser.add_argument('-s', '--shuffle', action='store_true',
                                help='Random shuffle the data', dest='shuffle')
        init_fuel_parser.add_argument('-n', '--normalize', action='store_true',
                                help='Normalize the data', dest='normalize')
        init_fuel_parser.add_argument('-p', '--prefix', action='store', type=str, default='',
                                help='The prefix of the file, default konkon', dest='prefix')
        init_fuel_parser.add_argument('-c', '--concat', action='store', type=int, default='',
                                help='The prefix of the file, default konkon', dest='concat',
                                nargs=2)

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

    def init_fuel(self, **kwargs):
        init_fuel.init(**kwargs)



def init_log_settings():
    logging.basicConfig(format='[%(asctime)s.%(msecs)d] %(module)s - %(levelname)s : %(message)s',
                        datefmt='%H:%M:%S',
                        level=LOG_LEVEL)


def main():
    init_log_settings()
    Main()


main()
