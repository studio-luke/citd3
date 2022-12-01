"""PyAudio Example: Play a wave file (callback version)."""

import pyaudio
import wave
import time
import sys


if __name__ == '__main__':
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 44100  # Record at 44100 samples per second
    seconds = 5
    filename = "output.wav"

    # audiofile = input("Audio File Name: ")
    # wf = wave.open(audiofile, 'rb')

    if False:
        audiofile = "../norabang_sample.wav"
        wf = wave.open(audiofile, 'rb')

        # instantiate PyAudio (1)
        p = pyaudio.PyAudio()

        # open stream (3)
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        data = wf.readframes(44100 * 10)
        stream.write(data)
        print("hi")

        # stop stream (4)
        stream.stop_stream()
        stream.close()

        # close PyAudio (5)
        p.terminate()



    if True: 
        # instantiate PyAudio (1)
        p = pyaudio.PyAudio()

        # define callback (2)
        def callback(in_data, frame_count, time_info, status):
            data = wf.readframes(frame_count)
            print("hi")
            return (data, pyaudio.paContinue)

        input('Recording Start (Press Enter) ')

        # open stream using callback (3)
        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)
                        #stream_callback=callback)

        frames = []  # Initialize array to store frames

        data = stream.read(44100 * 5)
        print("hihi")

        # Store data in chunks for 3 seconds
        for i in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)  # data: byte format
            frames.append(data)

        print(len(frames))
        # Stop and close the stream 
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface

        p.terminate()

        print('Finished recording')

        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()