from sklearn.metrics.pairwise import cosine_similarity

class IRSystem:
    ''' Information Retrieval System using TF-IDF and Cosine Similarity.'''
    
    def __init__(self, tfidf_matrix, df, text_column="cleaned_text"):
        self.tfidf_matrix = tfidf_matrix
        self.df = df
        self.text_column = text_column

    def query(self, text, vectorizer, top_n=5):
        """
        Returns top_n most similar complaints to the input text.
        """
        query_vec = vectorizer.transform([text])

        # 2. Compute cosine similarities against all complaints
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # 3. Sort indices by similarity (highest first)
        sorted_indices = similarities.argsort()[::-1]

        # 4. Take more than top_n initially so we have room to drop duplicates
        candidate_indices = sorted_indices[: min(top_n * 50, len(sorted_indices))]  # 5x buffer

        # 5. Build initial results frame
        results = self.df.iloc[candidate_indices].copy()
        results["similarity_score"] = similarities[candidate_indices]

        results = results.drop_duplicates()
        results = results.sort_values(by="similarity_score", ascending=False)

        # 7. Return only top_n unique results
        results = results.head(top_n)

        return results
