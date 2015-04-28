#!/usr/bin/env python

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)

from flask.ext.script import Manager, Server
from flask.ext.pymongo import PyMongo
from data_server import app

manager = Manager(app)
manager.add_command('runserver', Server(
    use_debugger = True,
    use_reloader = True,
    host = '0.0.0.0',
    port = '4999')
)

if __name__ == "__main__":
    manager.run()
