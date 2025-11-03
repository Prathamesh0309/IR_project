from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

class IRSystem:
    def __init__(self, tfidf_matrix, df, text_column="cleaned_text"):
        self.tfidf_matrix = tfidf_matrix
        self.df = df
        self.text_column = text_column

    def query(self, text, vectorizer, top_n=5):
        """
        Returns top_n most similar complaints to the input text.
        """
        query_vec = vectorizer.transform([text])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]
        results = self.df.iloc[top_indices].copy()
        results["similarity_score"] = similarities[top_indices]
        return results
