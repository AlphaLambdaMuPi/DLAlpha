import logging, logging.handlers
import sys
from settings import *
from check_file import check_file, load_shelve

def init():
    init_log_settings()
    check_file()
    load_shelve(True)
