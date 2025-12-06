from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class FeatureEngineer:
    def __init__(self, max_features=50000, ngram_range=(1,2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range,
                                          min_df=10, max_df=0.9, stop_words='english')

    def fit_transform(self, df, text_column="cleaned_text"):
        '''
        Fit the TF-IDF vectorizer to the specified text column and transform the data.
        '''
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
        '''
        Transform new data using the already fitted TF-IDF vectorizer.'''
        if df is None:
            raise ValueError("FeatureEngineer.transform: received None for `df`. Ensure data is loaded before calling transform.")
        if text_column not in df.columns:
            raise KeyError(f"FeatureEngineer.transform: text column '{text_column}' not found in DataFrame. Available columns: {list(df.columns)}")
        return self.vectorizer.transform(df[text_column])

    def get_feature_names(self):
        '''
        Get the feature names from the TF-IDF vectorizer.'''
        return self.vectorizer.get_feature_names_out()
    