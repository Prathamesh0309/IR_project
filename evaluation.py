from sklearn.metrics import precision_score, recall_score, f1_score

SMART_TEST_QUERIES = [
    "loud noise at night",
    "illegal parking issue",
    "trash not collected",
    "street light not working",
    "rats or pest problem",
]

def evaluate_with_pseudo_labels(ir, fe, df, top_k=20, rel_top_percent=0.1):
    '''
    Evaluate the IR system using pseudo-relevance labels based on TF-IDF similarity.
    For each query in SMART_TEST_QUERIES, considers the top X% most similar documents
    as relevant, and computes precision, recall, and F1-score based on the IR system'''
    
    text_candidates = ["cleaned_text", "full_text", "Descriptor"]
    text_col = next((c for c in text_candidates if c in df.columns), None)

    if text_col is None:
        raise KeyError(
            "No suitable text column found (expected one of cleaned_text, full_text, Descriptor)."
        )

    results = []
    total_docs = len(df)
    rel_cutoff = max(1, int(total_docs * rel_top_percent / 100))

    for query in SMART_TEST_QUERIES:

        # Vectorize query
        query_vec = fe.vectorizer.transform([query])

        # Vectorize all documents using the detected text column
        all_texts = df[text_col].astype(str)
        all_vecs = fe.vectorizer.transform(all_texts)

        # Compute similarity for all docs
        sims = (all_vecs @ query_vec.T).toarray().flatten()

        # Top X% = pseudo relevant
        relevant_true_indices = sims.argsort()[-rel_cutoff:]

        # Retrieved from system
        retrieved_df = ir.query(query, vectorizer=fe.vectorizer, top_n=top_k)
        retrieved_indices = retrieved_df.index.tolist()

        # Build complete set for P/R/F1
        all_indices = set(retrieved_indices) | set(relevant_true_indices)
        y_true = []
        y_pred = []

        for idx in all_indices:
            y_pred.append(1 if idx in retrieved_indices else 0)
            y_true.append(1 if idx in relevant_true_indices else 0)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            "query": query,
            "precision": round(float(precision), 3),
            "recall": round(float(recall), 3),
            "f1": round(float(f1), 3),
        })

    return results
