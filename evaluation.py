from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import silhouette_score

class Evaluator:
    @staticmethod
    def clustering_score(tfidf_matrix, labels):
        score = silhouette_score(tfidf_matrix, labels)
        print(f"Silhouette Score: {score:.3f}")
        return score

    @staticmethod
    def retrieval_score(y_true, y_pred):
        """
        y_true, y_pred should be lists of 0/1 indicating relevance
        """
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        return precision, recall, f1
