import motion
from multiprocessing import Process, Pipe

#import sound

def show(pipe):
    while True:
        print(pipe.recv())

if __name__ == "__main__":
    receive_pipe, send_pipe = Pipe(duplex=False)
    p1 = Process(target = motion.motion_main.main, args=(send_pipe,))
    p2 = Process(target = show, args=(receive_pipe,))
    p1.start()
    p2.start()