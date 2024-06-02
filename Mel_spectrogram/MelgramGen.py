import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

gpt_name = 'gpt-3.5-turbo-16k'

path = '/root/autodl-tmp/MVA/results/'+ gpt_name +'/'
path_save = '/root/autodl-tmp/MVA/Mel_spectrogram/results/'+ gpt_name +'/'
files= os.listdir(path)

for file in files:
    audio_data = path + file
    x , sr = librosa.load(audio_data)
    mel_spect = librosa.feature.melspectrogram(y=x, sr=sr)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    librosa.display.specshow(mel_spect, y_axis='mel', fmax=sr/2, x_axis='time')
    plt.title(file)
    #plt.colorbar(format='%+2.0f dB')
    plt.savefig(path_save + file + ".png")
