from pydub import AudioSegment
import collections
import contextlib
import sys
import wave
import webrtcvad
import os

#https://stackoverflow.com/questions/5120555/how-can-i-convert-a-wav-from-stereo-to-mono-in-python
def stereoToMono(input_path, output_path):
    sound = AudioSegment.from_wav(input_path)
    sound = sound.set_channels(1)
    sound.export(output_path, format="wav")


def main(args):
    source_path = args[0]
    output_path = args[1]

    files = [os.path.join(source_path, f)
        for f in os.listdir(source_path) if f.endswith('.wav')]

    for f in files:
        #tylko stereo
        with contextlib.closing(wave.open(f, 'rb')) as wf:
            num_channels = wf.getnchannels()
            if  num_channels != 2:
                print(f"plik jest juz mono, {f} przetwarzamy tylko stereo!!!")

            assert num_channels == 2


        filename = os.path.basename(f)
        output_file =  output_path + '/' + filename[:-4] + '_mono.wav'

        stereoToMono(f, output_file)


if __name__ == '__main__':
    main(sys.argv[1:])    