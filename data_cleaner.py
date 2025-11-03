import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

class DataCleaner:
    def __init__(self, text_column="Descriptor"):
        self.text_column = text_column
        # Attempt to ensure NLTK data is available; don't crash if downloads fail.
        try:
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)
        except Exception as e:
            print(f"Warning: NLTK download failed or unavailable: {e}")
        # Attempt to build stopwords; if not available, fall back to a small builtin set.
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception:
            print("Warning: NLTK stopwords not available; using small fallback set.")
            self.stop_words = set(["the", "and", "is", "in", "to", "of", "a"])
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", "", text)
        tokens = [self.lemmatizer.lemmatize(word) for word in text.split() if word not in self.stop_words]
        return " ".join(tokens)

    def preprocess(self, df):
        print("Cleaning text data...")
        df["cleaned_text"] = df[self.text_column].apply(self.clean_text)
        print("Text cleaning complete.")
        return df
