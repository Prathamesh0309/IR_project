from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class FeatureEngineer:
    def __init__(self, max_features=5000, ngram_range=(1,1)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)

    def fit_transform(self, df, text_column="cleaned_text"):
        # Defensive checks to provide clearer errors when input is invalid
        if df is None:
            raise ValueError("FeatureEngineer.fit_transform: received None for `df`. Ensure data is loaded before calling fit_transform.")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"FeatureEngineer.fit_transform: expected a pandas DataFrame, got {type(df)}")
        if text_column not in df.columns:
            raise KeyError(f"FeatureEngineer.fit_transform: text column '{text_column}' not found in DataFrame. Available columns: {list(df.columns)}")

        print("Creating TF-IDF matrix...")
        tfidf_matrix = self.vectorizer.fit_transform(df[text_column])
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        return tfidf_matrix

    def transform(self, df, text_column="cleaned_text"):
        if df is None:
            raise ValueError("FeatureEngineer.transform: received None for `df`. Ensure data is loaded before calling transform.")
        if text_column not in df.columns:
            raise KeyError(f"FeatureEngineer.transform: text column '{text_column}' not found in DataFrame. Available columns: {list(df.columns)}")
        return self.vectorizer.transform(df[text_column])

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()
    