from scipy.io import wavfile
from scipy.signal import resample, butter, lfilter
from tacotron2.prepro import TacoProcess
import numpy as np
import os
from tacotron2.hyperparams import Hyperparams as hp

class Preprocesser:
    def __init__(self, audio_path: str):
        self.audio = wavfile.read(audio_path)
        self.sample_rate = self.audio

    def resample(self) -> object:
        new_sample_rate = 16000
        resampled_audio = resample(self.audio, int(len(self.audio) * new_sample_rate / self.sample_rate))

        self.sample_rate = new_sample_rate
        return resampled_audio
    
    def filter(self) -> object:
        cutoff_frequency = 2000
        nyquist_frequency = 0.5 * self.sample_rate
        b, a = butter(4, cutoff_frequency / nyquist_frequency, 'low')

        return lfilter(b, a, self.audio)
    
    def taco_process(self, features: [], count: int):
        mel_spec = features[0]
        mag_spec = features[1]
        np.save(os.path.join(os.path.join(hp.data, 'mels'), f"{str(count)}.npy"), mel_spec)
        np.save(os.path.join(os.path.join(hp.data, 'mags'), f"{str(count)}.npy"), mag_spec)
        wavfile.write(os.path.join(os.path.join(hp.data, 'wavs'), f"{str(count)}.wav"), self.sample_rate, self.audio)
        processor = TacoProcess()
        processor.taco_process()
        print(f"File {str(count)} processed")