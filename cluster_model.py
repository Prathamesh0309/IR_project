from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import time
import random

class ClusterModel:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.model = None

    def kmeans(self, tfidf_matrix):
        # Default behavior: perform kmeans and optionally compute silhouette score
        print(" Performing KMeans clustering...")
        try:
            nnz = getattr(tfidf_matrix, 'nnz', None)
            print(f"Matrix shape: {tfidf_matrix.shape} ({nnz} non-zero entries)")
        except Exception:
            print(f"Matrix info unavailable; object: {type(tfidf_matrix)}")

        start_time = time.time()
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = self.model.fit_predict(tfidf_matrix)
        clustering_time = time.time() - start_time

        print(f" Clustering completed in {clustering_time:.1f}s")

        # Return labels and optionally let caller compute silhouette (which can be slow for large n)
        cluster_sizes = np.bincount(labels)
        for i in range(self.n_clusters):
            pct = 100 * cluster_sizes[i] / len(labels) if len(labels) > 0 else 0
            print(f"   Cluster {i}: {cluster_sizes[i]} items ({pct:.1f}%)")

        return labels

    def hierarchical(self, tfidf_matrix):
        print("🤖 Performing Hierarchical clustering...")
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters)
        labels = self.model.fit_predict(tfidf_matrix.toarray())
        print("✅ Hierarchical clustering done")
        return labels

    def visualize_clusters_wordcloud(self, df, labels, text_column="cleaned_text",max_words=100):
        df['cluster'] = labels
        n_clusters = len(set(labels))
    
        for i in range(n_clusters):
            cluster_df = df[df['cluster'] == i]
            # Sample up to 2000 texts to keep it fast
            sample_texts = cluster_df[text_column].dropna().sample(min(2000, len(cluster_df)), random_state=42)
            text = " ".join(sample_texts)

            # Generate the word cloud
            wc = WordCloud(width=800, height=400, background_color='white', max_words=max_words).generate(text)

            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"Word Cloud for Cluster {i}")
            plt.show()

    def silhouette_analysis(self, tfidf_matrix, k_min=2, k_max=10,sample_size=10000):
        """
        Compute and plot silhouette scores for a range of k values.
        Only use this when visualize=True as a one-time analysis.
        """
        if tfidf_matrix.shape[0] > sample_size:
            idx = np.random.choice(tfidf_matrix.shape[0], sample_size, replace=False)
            sample_matrix = tfidf_matrix[idx]
            print(f"Sampling {sample_size} rows for silhouette analysis")
        else:
            sample_matrix = tfidf_matrix

        scores = []
        ks = range(k_min, k_max + 1)

        for k in ks:
            clusterer = KMeans(n_clusters=k, random_state=42)
            labels = clusterer.fit_predict(sample_matrix)
            score = silhouette_score(sample_matrix, labels)
            scores.append(score)
            print(f"   k={k}: silhouette score={score:.4f}")

        plt.figure(figsize=(6,4))
        plt.plot(ks, scores, marker='o', color='darkorange')
        plt.title("Silhouette Score vs Number of Clusters")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Average Silhouette Score")
        plt.grid(True)
        plt.show()


    def plot_top_complaints(self, df, text_column="cleaned_text", top_n=10):
        """
        Plots top N complaint categories in the dataset as a bar chart.
        """
        top_complaints = df[text_column].value_counts().head(top_n)
        plt.figure(figsize=(8,5))
        top_complaints.plot(kind='bar', color='skyblue')
        plt.title(f"Top {top_n} Complaint Types")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()  

    def plot_clusters_2d(self, tfidf_matrix, labels):
        """
        Reduces TF-IDF to 2D using PCA and plots clusters.
        """
        print("🔹 Reducing dimensions for 2D cluster visualization...")
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(tfidf_matrix.toarray())
        plt.figure(figsize=(6,5))
        plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap='viridis', s=10)
        plt.title("2D PCA Cluster Visualization")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.show()

    def find_optimal_clusters(self, tfidf_matrix, max_k=10, sample_size=50000):
        """
        Determine optimal number of clusters using silhouette and elbow methods.
        Works on a sample of the data for speed.
        """
        print("\n🔍 Finding optimal number of clusters (this may take several minutes)...")

        # Subsample for speed (only if dataset is very large)
        if tfidf_matrix.shape[0] > sample_size:
            sample_idx = random.sample(range(tfidf_matrix.shape[0]), sample_size)
            tfidf_sample = tfidf_matrix[sample_idx]
        else:
            tfidf_sample = tfidf_matrix

        sil_scores = []
        inertias = []
        K = range(2, max_k + 1)

        for k in K:
            print(f"  Evaluating k={k} ...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(tfidf_sample)
            inertias.append(kmeans.inertia_)

            # Compute silhouette only if it won’t blow up memory
            try:
                score = silhouette_score(tfidf_sample, labels)
                sil_scores.append(score)
            except Exception as e:
                print(f"    ⚠️ Skipping silhouette for k={k}: {e}")
                sil_scores.append(None)

        # Plot both metrics side by side
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(K, inertias, 'bx-', label='Inertia (Elbow)')
        ax1.set_xlabel('k')
        ax1.set_ylabel('Inertia', color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot(K, sil_scores, 'ro-', label='Silhouette Score')
        ax2.set_ylabel('Silhouette Score', color='r')
        ax2.tick_params('y', colors='r')

        plt.title('Optimal Cluster Evaluation (Elbow + Silhouette)')
        plt.show()

        # Print top silhouette
        best_k = K[np.nanargmax(sil_scores)]
        print(f"\n✅ Best k based on silhouette score: {best_k} (Score={max(sil_scores):.4f})")

        return best_k