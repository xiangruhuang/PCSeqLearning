import time

class Timer(object):
    def __init__(self, name='default'):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        self.end = time.time()
        self.time_elapsed = time.time()-self.start
        print(f'Elapsed Time in {self.name} = {self.time_elapsed:.4f} s')

if __name__ == '__main__':
    with Timer('hey'):
        for i in range(20000000):
            j = i*i
