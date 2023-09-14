import subprocess
from text_analysis import TextAnalyser
from feature_extraction import FeatureExtractor
from preprocess import Preprocesser
from tacotron2.hyperparams import Hyperparams as hp
import os

def process(audio_filename: str, count: int):
    feature_extractor = FeatureExtractor(audio_filename)
    features = feature_extractor.extract_features()

    processor = Preprocesser('tacotron2/' + hp.data + '/wavs')
    processor.taco_process(features, count)

def main(train_loc: str):
    count = 0
    for file in os.listdir("audio_files"):
        count += 1
        process(file, count)
    
    print("All files processed")

    script = ["python", train_loc, "--output_directory='output'", "--log_directory='log'"]
    try:
        subprocess.run(script, check=True)
        print("Model trained successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error training model: {e}")

if __name__ == '__main__':
    main()