from time import time, process_time

class Timer:
    def __init__(self):
        self.time = time()
        self.python = process_time()
        self.clocks = {}
        self.active = {}

    def start(self, s):
        if s not in self.clocks:
            self.clocks[s] = -process_time()
        else:
            self.clocks[s] -= process_time()

        self.active[s] = True

    def stop(self, s):
        if s not in self.clocks:
            raise ValueError('{} not in clocks'.format(s))

        if not self.active[s]: return
        self.clocks[s] += process_time()
        self.active[s] = False

    def get(self, s = None):
        if s is None:
            return time() - self.time

        if s == 'python':
            return process_time() - self.python

        if s not in self.clocks:
            raise ValueError('{} not in clocks'.format(s))

        if self.active[s]:
            return self.clocks[s] + process_time()
        else:
            return self.clocks[s]
