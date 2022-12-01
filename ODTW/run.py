from scipy.io.wavfile import read as wfread
import scipy
import pyaudio
import wave
import librosa
import libfmp
import numpy as np
import matplotlib.pyplot as plt
from OEM import ODTW
import time
import sounddevice as sd

def compute_optimal_warping_path(D):
    """Compute the warping path given an accumulated cost matrix

    Notebook: C3/C3S2_DTWbasic.ipynb

    Args:
        D (np.ndarray): Accumulated cost matrix

    Returns:
        P (np.ndarray): Optimal warping path
    """
    N = D.shape[0]
    M = D.shape[1]
    n = 0
    m = 0
    P = [(n, m)]
    while n < N - 1 and m < M - 1:
        #if n == 0:
        #    cell = (0, m - 1)
        #elif m == 0:
        #    cell = (n - 1, 0)
        #else:
        val = min(D[n+1, m+1], D[n+1, m], D[n, m+1])
        if val == D[n+1, m+1]:
            cell = (n+1, m+1)
        elif val == D[n+1, m]:
            cell = (n+1, m)
        else:
            cell = (n, m+1)
        P.append(cell)
        (n, m) = cell
    # P.reverse()
    return np.array(P)

sample_name = "norabang_sample.wav"
chunk= 4409  # I don't know why but if chunk=4410, SFTF with hop size 2205 makes 3 time output 
audio_format=pyaudio.paFloat32
timeunit = 0.1
rate=44100
channels=1
sd.default.channels = 1

acc_name = "norabang_instruments.wav"

# Output PyAudio Stream for playing accompaniment
outp = pyaudio.PyAudio()
acc_audio = wave.open(acc_name, "rb")
accompaniment = outp.open(format=outp.get_format_from_width(acc_audio.getsampwidth()),
                channels=acc_audio.getnchannels(),
                rate=acc_audio.getframerate(),
                output=True)



# Open Input PyAudio Stream for Recording
inp = pyaudio.PyAudio()

stream = inp.open(format=audio_format,
        channels=channels,
        rate=rate,
        frames_per_buffer=chunk,
        input=True)

frames = []  # Initialize array to store frames

# t, j = 1, 1
# previous = None
# path = []

N, H = 4410, 2205

# Open Sample Audio for Comparing with Recorded Audio
sample_rate, sample_data = wfread(sample_name)

if sample_data.dtype != np.float32:
    X_float32 = sample_data.astype(np.float32)
    if sample_data.dtype == np.int16:
        max_int = 2**15
    else:
        assert False, "dtype was"+str(sample_data.dtype)

    X_stereo = X_float32 / max_int
    
if X_stereo.shape[1] == 1:  # Mono
    pass
else:   # Convert Stereo to Mono
    X_arr = np.array([(X_stereo[x][0] + X_stereo[x][1]) / 2 
                      for x in range(X_stereo.shape[0])])


# STFT on X
X = librosa.feature.chroma_stft(y=X_arr, sr=rate, tuning=0, norm=2, hop_length=H, n_fft=N)
scenario = ODTW(w=0.1, R=X, dist='euclidean')
print(X.shape)

# Read chunk from sample audio for oDTW initialization
Y_arr = sd.rec(int(timeunit * rate), rate, blocking=True)

"""
cnt = 0
for i in range(Y_arr.shape[0]):
    if (Y_arr[i] == np.nan):
        cnt+=1

print(Y_arr, "nan:", cnt)
"""

# Y_byte = stream.read(chunk)
# Y_int = np.frombuffer(Y_byte, dtype=np.int32)
# Y_float32 = Y_int.astype(np.float32)
# max_int = 2**31
# Y_arr = Y_float32 / max_int
# print("Y_flat32.shape:", Y_float32.shape)


Y = librosa.feature.chroma_stft(y=Y_arr, sr=rate, tuning=0, norm=2, hop_length=H, n_fft=N)
print(Y.shape)

# Initial accompaniment play
data = acc_audio.readframes(chunk)
data = np.frombuffer(data, dtype=np.int32)
sd.play(data)
#accompaniment.write(data)

# oDTW initialization
XY = scenario.init_dist(Y)
D = libfmp.c3.compute_accumulated_cost_matrix(XY)
P = compute_optimal_warping_path(D)
print(XY.shape)
cur = P[-1][0]  # Latest position of oDTW Alignment

sec = 1
duration = 15  # seconds
print("cur:",cur)
start_time = time.time()
loop = 0
while sec < duration:
    loop_start_time = time.time()
    Y_arr = sd.rec(int(timeunit * rate), rate)
    # Y_byte = stream.read(chunk)  # data: byte format
    # Y_arr = np.frombuffer(Y_byte, dtype=np.float32) # byte to NParray
    Y = librosa.feature.chroma_stft(y=Y_arr, sr=rate, tuning=0, norm=2, hop_length=H, n_fft=N)
    col = scenario.update_dist(Y)
    XY = np.column_stack((XY, col))
    
    # Update Accumulated Cost Matrix
    D = np.c_[D, np.zeros(D.shape[0])]
    D[0, -1] = (D[0,-2] + XY[0,-1])
    for i in range(1, X.shape[1]):
        D[i][-1] = (XY[i,-1] + min(D[i-1,-1], D[i, -2], D[i-1, -2]))
    
    # Update cur variable
    next_path = min(D[cur+1,-2], D[cur+1, -1], D[cur, -1])
    
    while True:
        if next_path == D[cur+1, -1]:
            cur += 1
            data = acc_audio.readframes(chunk)
            data = np.frombuffer(data, dtype=np.int32)
            sd.play(data, rate)
            # accompaniment.write(data)
        elif next_path == D[cur, -1]:
            pass
        else:
            cur += 1
            data = acc_audio.readframes(chunk)
            data = np.frombuffer(data, dtype=np.int32)
            sd.play(data, rate)
            # accompaniment.write(data)
            next_path = min(D[cur+1,-2], D[cur+1, -1], D[cur, -1])
            continue
        break
    
    # Save Recorded Audio
    frames.append(Y_byte)

    recent_time = time.time()
    # chunk / sample_rate ~= 0.1, so each loop should take 0.1s
    loop += 1

    if (recent_time - start_time >= sec):
        print("sec: ", recent_time - start_time, "cur:", cur, "processed sec:", loop * chunk / rate)
        sec += 1


accompaniment.stop_stream()
accompaniment.close()
outp.terminate()

        
P = compute_optimal_warping_path(D)

plt.imshow(XY, vmin=0, vmax=3)
plt.title('Recursive Matrix')
#plt.xticks(np.arange(XY.shape[1])+0.5)
#plt.yticks(np.arange(XY.shape[0])+0.5)
#plt.ylim([-0.5,9.5])
#plt.xlim([0.5+i, 10.5+i])
plt.grid(True, linestyle='--')
plt.colorbar()
plt.show()

libfmp.c3.plot_matrix_with_points(D, P, linestyle='-', marker='', 
    aspect='equal', clim=[0, 300], 
    title='$D$ with optimal warping path', xlabel='Sequence Y', ylabel='Sequence X');

print(X_arr.shape, X.shape)
print(Y_arr.shape, Y.shape)

stream.stop_stream()
stream.close()
inp.terminate()

