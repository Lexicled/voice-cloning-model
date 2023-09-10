from scipy.io import wavfile
from scipy.signal import resample, butter, lfilter

class Preprocesser:
    def __init__(self, audio_path: str):
        self.audio = wavfile.read(audio_path)
        self.self.sample_rate = self.audio

    def resample(self) -> object:
        new_sample_rate = 16000
        resampled_audio = resample(self.audio, int(len(self.audio) * new_sample_rate / self.sample_rate))

        self.self.sample_rate = new_sample_rate
        return resampled_audio
    
    def filter(self) -> object:
        cutoff_frequency = 2000
        nyquist_frequency = 0.5 * self.self.sample_rate
        b, a = butter(4, cutoff_frequency / nyquist_frequency, 'low')

        return lfilter(b, a, self.audio)