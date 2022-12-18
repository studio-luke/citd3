import numpy as np
import librosa
import time
import pyaudio
import math
import keyboard

import motion
from multiprocessing import Process, Queue

def distance(x, y):
    total = 0.0
    for i in range(12):
        total = total + (x[i] - y[i]) * (x[i] - y[i])
    return math.sqrt(total)

if __name__ == '__main__':
    q = Queue()

    process = Process(target=motion.motion_main.main,args=(q,))
    process.start()


    inst, inst_sr = librosa.load("wav/inst.wav", dtype="float32", sr=None)
    # inst_sr = 48000

    INST_RATE = inst_sr
    INST_FRAME = int(INST_RATE / 10)

    vocal, vocal_sr = librosa.load("wav/vocal.wav", dtype="float32", sr=None)
    # vocal_sr = 48000

    VOCAL_RATE = vocal_sr
    VOCAL_FRAME = int(VOCAL_RATE / 10)

    chroma_ref = librosa.feature.chroma_stft(
        y=vocal, sr=vocal_sr, tuning=0, norm=2, hop_length=VOCAL_FRAME, center=False
    )


    INPUT_RATE = VOCAL_RATE
    INPUT_FRAME = int(INPUT_RATE / 10)

    input_half, dummy = librosa.load("wav/vocal_05x.wav", dtype="float32", sr=None)
    input_one = vocal
    input_one_half, dummy = librosa.load("wav/vocal_15x.wav", dtype="float32", sr=None)
    input_twice, dummy = librosa.load("wav/vocal_2x.wav", dtype="float32", sr=None)

    input = input_twice

    print("loaded")

    is_out_ended = False

    VERY_BIG = 999.0
    MAX = 5

    WIN_SIZE = 50

    D = np.full((chroma_ref.shape[1], 3 * chroma_ref.shape[1]), VERY_BIG)

    i_frame = 0
    t = h = 0
    path_t = []
    path_h = []
    STEP = 8
    is_mic = False


    def callback(in_data, frame_count, time_info, flag):
        # in_data : 4 bytes
        global i_frame, t, h, D, path_t, path_h, is_out_ended
        print("call back: ", i_frame)

        # input part

        if i_frame == 0:
            if is_mic == False:
                tmp = input[0:frame_count]
            else:
                tmp = np.frombuffer(in_data, np.float32)
            chroma = librosa.feature.chroma_stft(
                y=tmp,
                sr=INPUT_RATE,
                norm=2,
                tuning=0,
                hop_length=frame_count,
                win_length=frame_count,
                n_fft=frame_count,
                center=False,
            )

            D[0][0] = distance(chroma_ref[:, [0]], chroma)

        else:
            if is_mic == False:
                tmp = input[i_frame * frame_count : (i_frame + 1) * frame_count]
            else:
                tmp = np.frombuffer(in_data, np.float32)

            chroma = librosa.feature.chroma_stft(
                y=tmp,
                sr=INPUT_RATE,
                norm=2,
                tuning=0,
                hop_length=frame_count,
                win_length=frame_count,
                n_fft=frame_count,
                center=False,
            )

            for i in range(max(1, h - WIN_SIZE), min(D.shape[0], h + WIN_SIZE)):
                dist = distance(chroma_ref[:, [i]], chroma)
                top = D[i][i_frame - 1]
                mid = D[i - 1][i_frame - 1]
                bot = D[i - 1][i_frame]
                if (top < mid) and (top < bot):
                    D[i][i_frame] = top + dist * 0.1
                elif mid < bot:
                    D[i][i_frame] = mid + dist * 0.11
                else:
                    D[i][i_frame] = bot + dist * 0.1

            cnt = 0
            while (cnt < MAX) and (t < i_frame) and (h < D.shape[0] - 1):
                up = D[h + 1][t]
                diag = D[h + 1][t + 1]
                side = D[h][t + 1]
                if (up < diag) and (up < side):
                    h = h + 1
                    cnt = cnt + 1
                elif diag < side:
                    h = h + 1
                    t = t + 1
                else:
                    t = t + 1
            if cnt == MAX:
                t = t + 1
            if not q.empty():
                if q.get() == 'Up':
                    print('up')
                    h += 1

        path_t = np.append(path_t, t)
        path_h = np.append(path_h, h)

        print(t, h)

        # output part

        if i_frame == 0:
            if is_mic == True:
                out_data = inst[:frame_count]
            else:
                out_data = inst[:frame_count] + input[:frame_count]
        else:
            if i_frame < STEP:
                pre = int(path_h[i_frame - 1])
                now = int(path_h[i_frame])
                slope = now - pre
            else:
                pre = int(path_h[i_frame - STEP])
                now = int(path_h[i_frame])
                slope = int((now - pre) / STEP)

            if slope >= 1:
                if (now + 1) * frame_count >= inst.shape[0]:
                    if is_mic == False:
                        out_data = input[i_frame * frame_count : -1]
                        out_data = np.pad(
                            out_data, frame_count - len(out_data), constant_values=0
                        )
                    else:
                        out_data = np.zeros(frame_count, dtype=np.float32)

                    tmp = inst[now * frame_count : -1]
                    tmp = np.pad(tmp, slope * frame_count - len(tmp), constant_values=0)
                    is_out_ended = True
                    next_status = pyaudio.paComplete
                else:
                    if is_mic == False:
                        out_data = input[
                            i_frame * frame_count : (i_frame + 1) * frame_count
                        ]
                    else:
                        out_data = np.zeros(frame_count, dtype=np.float32)
                    tmp = inst[now * frame_count : (now + slope) * frame_count]

                out_data = out_data + librosa.effects.time_stretch(tmp, rate=slope)

            elif slope == 0:
                if is_mic == False:
                    out_data = input[i_frame * frame_count : (i_frame + 1) * frame_count]
                else:
                    out_data = np.zeros(frame_count, dtype=np.float32)

        i_frame = i_frame + 1

        if is_out_ended:
            next_status = pyaudio.paComplete
        else:
            next_status = pyaudio.paContinue
        return (out_data, next_status)


    p = pyaudio.PyAudio()

    FORMAT = pyaudio.paFloat32
    CHANNELS = 1

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=INPUT_RATE,
        input=True,
        output=True,
        frames_per_buffer=INPUT_FRAME,
        stream_callback=callback,
    )

    stream.stop_stream()

    i_frame = 0
    t = 0
    h = 0
    path_t = []
    path_h = []

    q.get()

    stream.start_stream()
    while True:
        if keyboard.is_pressed('q'):
            stream.stop_stream()
            break
        elif keyboard.is_pressed('r'):
            print("reset")
            stream.stop_stream()
            i_frame = 0
            t = 0
            h = 0
            path_t = []
            path_h = []
            stream.start_stream()
        elif keyboard.is_pressed('1'):
            print("0.5x")
            stream.stop_stream()
            is_mic = False
            input = input_half
            i_frame = 0
            t = 0
            h = 0
            path_t = []
            path_h = []
            stream.start_stream()
        elif keyboard.is_pressed('2'):
            print("1x")
            stream.stop_stream()
            is_mic = False
            input = input_one
            i_frame = 0
            t = 0
            h = 0
            path_t = []
            path_h = []
            stream.start_stream()
        elif keyboard.is_pressed('3'):
            print("1.5x")
            stream.stop_stream()
            is_mic = False
            input = input_one_half
            i_frame = 0
            t = 0
            h = 0
            path_t = []
            path_h = []
            stream.start_stream()
        elif keyboard.is_pressed('4'):
            print("2x")
            stream.stop_stream()
            is_mic = False
            input = input_twice
            i_frame = 0
            t = 0
            h = 0
            path_t = []
            path_h = []
            stream.start_stream()
        elif keyboard.is_pressed('5'):
            print("mic")
            stream.stop_stream()
            is_mic = True
            i_frame = 0
            t = 0
            h = 0
            path_t = []
            path_h = []
            stream.start_stream()
        if not q.empty():
            print(q.get())

    print("quit")
    stream.close()
    p.terminate()
    process.terminate()
