import sounddevice as sd
import numpy as np
import wave


audiofile = "../norabang_sample.wav"
wf = wave.open(audiofile, 'rb')

data = wf.readframes(22050 * 10)

X = np.frombuffer(data, dtype=np.int32)

sd.play(X, samplerate=44100, blocking=False)

print("hi")


data = wf.readframes(22050 * 10)
data = wf.readframes(22050 * 10)

print(type(sd.get_status()))
input()

X = np.frombuffer(data, dtype=np.int32)

sd.play(X, samplerate=44100, blocking=False)

print("hi2")

input()