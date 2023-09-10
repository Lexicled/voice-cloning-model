import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TextAnalyser:
    def __init__(self, text: str):
        self.text = text
    def extract_tokens(self):
        nltk.download('stopwords')
        nltk.download('punkt')

        self.text = self.text.lower()  # Lowercasing
        tokens = word_tokenize(self.text)  # Tokenization
        return [word for word in tokens if word not in stopwords.words('english')]