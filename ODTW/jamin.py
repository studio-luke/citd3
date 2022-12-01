import pyaudio
import numpy as np
#import struct
import time
import librosa
#import pyrubberband.pyrb as pyrb

FS = 44100
FRAME_SIZE = 4410
IDEAL_FREQ = 255.0
VOLUME = 0.1
CHANNELS = 1

x, rs = librosa.load('norabang_instruments.wav',dtype='float32', sr=None)

cur1 = i = 0
width = FRAME_SIZE
k = 1
def callback(in_data, frame_count, time_info, status):
    global cur1, i, width, k
    sec = cur1 / FS
    if(sec>5):
        k=0.5

    next_status = pyaudio.paContinue

    #ver1
    cur2 = int(cur1+width*k)
    if(cur2 >= x.shape[0]):
        cur2 = -1
        next_status = pyaudio.paComplete

    #ver2
    #cur2 = -1


    tmp = x[cur1:cur2]

    #mp1 = pyrb.time_stretch(tmp,sr=FS,rate=k)

    tmp1 = librosa.effects.time_stretch(tmp,rate=k)

    out_data = tmp1[0:FRAME_SIZE].astype(np.float32).tobytes()

    i = i + 1

    cur1 = cur1 + int(width*k)

    return (out_data, next_status)

p=pyaudio.PyAudio()

stream = p.open(format = pyaudio.paFloat32,
               channels =1,
               rate=FS,
               output=True,
                frames_per_buffer=FRAME_SIZE,
               stream_callback=callback)

stream.start_stream()

for j in range(10):
    time.sleep(1)
    print(j)

stream.stop_stream()
stream.close()

p.terminate()
