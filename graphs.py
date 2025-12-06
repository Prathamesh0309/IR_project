import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud

# ---------------------------
# 1. Load NYC 311 dataset
# ---------------------------
# Replace 'path_to_file.csv' with your Kaggle cached dataset path

df = pd.read_csv(r"data/raw_311_data.csv")

# Inspect the columns
print(df.columns)
# Usually the complaint description column is named 'Descriptor' or 'Complaint Type'
text_column = "Descriptor"  # adjust if different

# ---------------------------
# 2. Top complaint categories (bar chart)
# ---------------------------
top_complaints = df[text_column].value_counts().head(10)
plt.figure(figsize=(8,5))
top_complaints.plot(kind='bar', color='skyblue')
plt.title("Top 10 NYC 311 Complaint Types")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ---------------------------
# 3. TF-IDF vectorization & K-Means clustering
# ---------------------------
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
X = tfidf.fit_transform(df[text_column].astype(str))

# Example: 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Optional: Inspect top terms per cluster
terms = tfidf.get_feature_names_out()
for i in range(5):
    print(f"\nCluster {i} top terms:")
    cluster_center = kmeans.cluster_centers_[i]
    top_terms_idx = cluster_center.argsort()[-10:][::-1]
    print([terms[j] for j in top_terms_idx])

# ---------------------------
# 4. Word clouds for clusters
# ---------------------------
for i in range(5):
    cluster_texts = df[df['cluster']==i][text_column].astype(str)
    text = " ".join(cluster_texts)
    wc = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for Cluster {i}")
    plt.show()

# ---------------------------
# 5. Optional: 2D scatter plot (if you reduce dimensions)
# ---------------------------
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X.toarray())

plt.figure(figsize=(6,5))
plt.scatter(reduced_X[:,0], reduced_X[:,1], c=df['cluster'], cmap='viridis', s=10)
plt.title("K-Means Clusters (2D PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
