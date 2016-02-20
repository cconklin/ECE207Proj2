import time

driver = None

class Timer:    
    def __enter__(self):
        self.start = driver.Event()
        self.end = driver.Event()
        self.start.record()
        return self

    def __exit__(self, *args):
        self.end.record()
        self.end.synchronize()
        self.interval = self.start.time_till(self.end)*1e-3
