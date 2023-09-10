import subprocess
from text_analysis import TextAnalyser
from feature_extraction import FeatureExtractor

def main(audio_filename: str, train_loc: str):
    feature_extractor = FeatureExtractor(audio_filename)
    features = feature_extractor.extract_features()

    script = ["python", train_loc, "--output_directory='output'", "--log_directory='log'"]

    try:
        subprocess.run(script, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error training model: {e}")

if __name__ == '__main__':
    main()