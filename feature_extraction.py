import librosa
from preprocess import Preprocesser

class FeatureExtractor:
    def __init__(self, filename):
        self.filename = filename

    def preprocess(self) -> object:
        preprocessor = Preprocesser(self.filename)
        return preprocessor.filter(preprocessor.resample())

    def extract_features(self) -> []:
        audio_file = self.preprocess(self.filename)
        audio, sample_rate = librosa.load(audio_file)
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        chroma = librosa.feature.chroma_cens(y=audio, sr=sample_rate)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)

        return [mfccs, chroma, spectral_contrast, spectral_centroid, zero_crossing_rate]