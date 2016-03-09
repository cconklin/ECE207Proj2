import time

class GPUTimer(object):
    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        self.start = self.driver.Event()
        self.end = self.driver.Event()
        self.start.record()
        return self

    def __exit__(self, *args):
        self.end.record()
        self.end.synchronize()
        self.interval = self.start.time_till(self.end)

    def __str__(self):
        return "{:.3f} ms".format(self.interval)

class Timer(object):
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

    def __str__(self):
        return "{:.3f} ms".format(self.interval * 1000)
