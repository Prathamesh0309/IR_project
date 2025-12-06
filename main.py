from kaggle_data_download import DataDownloader
from data_cleaner import DataCleaner
from feature_engineering import FeatureEngineer
from cluster_model import ClusterModel
from topic_model import TopicModel
from exploratory_analysis import ExploratoryAnalysis
from retrieval_system import IRSystem
from recommendation import Recommender
import os
import joblib
import json
from scipy import sparse


def get_relevant_ids(df, keyword, top_k=20):
    """
    Automatically find relevant complaint IDs by searching for keywords
    in cleaned_text column.
    """
    mask = df["cleaned_text"].str.contains(keyword, case=False, na=False)
    return df.loc[mask, "unique_key"].head(top_k).tolist()

def build_test_query_set(df):
    """
    Build evaluation test cases using real complaint IDs from df.
    """
    return [
        {
            "query": "loud noise at night",
            "relevant_ids": get_relevant_ids(df, "noise")
        },
        {
            "query": "illegal parking",
            "relevant_ids": get_relevant_ids(df, "parking")
        },
        {
            "query": "rats in basement",
            "relevant_ids": get_relevant_ids(df, "rat")
        }
    ]



def download_and_load():
    '''
    Download the dataset from Kaggle (if not already present) and load into a DataFrame.'''
    downloader = DataDownloader(local_path="data/311-service-requests-from-2010-to-present.csv")
    return downloader.load_dataframe()


def clean_data(df, text_column="Descriptor"):
    # Auto-detect a sensible text column if the provided one is not present
    if df is None:
        raise ValueError("clean_data: received None for df")

    available = [c for c in df.columns]
    # normalize column names for matching
    cols_lower = {c.lower(): c for c in available}

    preferred = [text_column, 'detailed_description', 'brief_description', 'description', 'descriptor', 'complaint_type', 'complaint_description']
    chosen = None
    for p in preferred:
        if p is None:
            continue
        if p in cols_lower:
            chosen = cols_lower[p]
            break

    if chosen is None:
        # fallback to the first text-like column if any
        for c in available:
            if 'description' in c.lower() or 'desc' in c.lower() or 'complaint' in c.lower():
                chosen = c
                break

    if chosen is None:
        raise KeyError(f"No suitable text column found in DataFrame. Available columns: {available}")

    cleaner = DataCleaner(text_column=chosen)
    print(f"Using text column: {chosen} for cleaning")
    return cleaner.preprocess(df)


def save_cleaned(df, path="data/cleaned_311_data.pkl"):
    ''' Save cleaned dataframe to a pickle file.'''
    df.to_pickle(path)


def prepare_features(df, max_features=50000):
    ''' Prepare TF-IDF features from cleaned text data.'''
    fe = FeatureEngineer(max_features=max_features)
    tfidf_matrix = fe.fit_transform(df)
    return fe, tfidf_matrix

# --- New functions for clustering, topic modeling, EDA, IR, and recommendation -- #
def run_clustering(tfidf_matrix, df, n_clusters=5, visualize=False):
    '''
    Run KMeans clustering on the TF-IDF matrix and optionally visualize results.
    '''
    clusterer = ClusterModel(n_clusters=n_clusters)
    if visualize:
        # Determine optimal cluster number before running final kmeans
        best_k = clusterer.find_optimal_clusters(tfidf_matrix, max_k=10, sample_size=50000)
        clusterer.n_clusters = best_k
    labels = clusterer.kmeans(tfidf_matrix)
    df['cluster'] = labels

    # Visualizations
    if visualize:
        try:
            #silhouette analysis
            clusterer.silhouette_analysis(tfidf_matrix, k_min=2, k_max=10)
            # Word clouds
            clusterer.visualize_clusters_wordcloud(df, labels)
            # Top complaints bar chart
            clusterer.plot_top_complaints(df, text_column='cleaned_text', top_n=10)
            # 2D PCA scatter plot
            clusterer.plot_clusters_2d(tfidf_matrix, labels)
        except Exception as e:
            print(f"Warning: Could not generate word clouds: {str(e)}")
    return clusterer, labels


def run_topic_model(df, num_topics=5):
    ''' Run LDA topic modeling on the cleaned text data.'''

    # Prepare and train topic model
    tm = TopicModel(num_topics=num_topics)
    tm.prepare_corpus(df)
    tm.train_lda()
    try:
        tm.show_topics()
    except Exception:
        pass
    return tm

def exploratory_analysis(df):
    ''' Perform exploratory data analysis on the dataframe.'''
    ea = ExploratoryAnalysis(df)
    ea.convert_dates(date_column="Created Date")
    return ea


def run_ir_and_recommend(tfidf_matrix, df, fe, recommender, sample_query="loud noise at night", top_n=5):
    ''' Run information retrieval system and get recommendations.'''

    print("\nRunning Information Retrieval System...")
    ir = IRSystem(tfidf_matrix, df)
    results = ir.query(sample_query, vectorizer=fe.vectorizer, top_n=top_n)
    print("\nAttaching Recommendations...")
    for idx, row in results.iterrows():
        cluster_label = int(row["cluster"])
        rec_text = recommender.recommend(cluster_label)
        results.loc[idx, "recommendation"] = rec_text

    print(results[["Descriptor", "cluster", "recommendation"]])

    return ir, recommender, results



#-------------------------------------------------------------#
# Full pipeline function
#-------------------------------------------------------------#

def run_pipeline(save_cleaned_file=True, cleaned_path="data/cleaned_311_data.pkl", max_features=5000, n_clusters=5, num_topics=5, visualize=False):
    """
    Runs the full pipeline and returns a dict of important artifacts.
    """
    # Step 1: Download and load data
    df = download_and_load()
    # Downsample dataset
    df = df.sample(n=1000000, random_state=42).reset_index(drop=True)
    
    # Step 2: Clean text
    df = clean_data(df)
    print("--------------")
    # Step 3: Save cleaned data
    if save_cleaned_file:
        save_cleaned(df, path=cleaned_path)

    # Step 4: TF-IDF vectorization
    fe, tfidf_matrix = prepare_features(df, max_features=max_features)

    # Persist vectorizer and tfidf matrix to disk for app reuse
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    vec_path = os.path.join(os.path.dirname(cleaned_path), "vectorizer.joblib")
    tfidf_path = os.path.join(os.path.dirname(cleaned_path), "tfidf.npz")
    manifest_path = os.path.join(os.path.dirname(cleaned_path), "manifest.json")
    try:
        joblib.dump(fe.vectorizer, vec_path)
        sparse.save_npz(tfidf_path, tfidf_matrix)
        manifest = {
            "df_shape": list(getattr(df, 'shape', [])),
            "text_column": getattr(fe, 'text_column', None) if hasattr(fe, 'text_column') else None,
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        print(f"Saved vectorizer to {vec_path} and tfidf to {tfidf_path}")
    except Exception as e:
        print(f"Warning: could not persist TF-IDF artifacts: {e}")

    # Step 5: Clustering
    clusterer, labels = run_clustering(tfidf_matrix, df, n_clusters=n_clusters, visualize=visualize)
    df["cluster"] = labels

    cluster_names = {
    0: "Noise Issues",
    1: "Parking Problems",
    2: "Sanitation / Garbage Issues",
    3: "Street / Traffic Conditions",
    4: "Rodent / Pest Problems",
}

    recommender = Recommender(df, cluster_names=cluster_names)
    print("Recommender initialized.")

    # Step 6: Topic modeling
    tm = run_topic_model(df, num_topics=num_topics)

    # Step 7: Exploratory analysis
    ea = exploratory_analysis(df)

    # Week 5: IR Retrieval System (sample)
    ir, recommender, results = run_ir_and_recommend(tfidf_matrix, df, fe,recommender)

    return {
        "df": df,
        "fe": fe,
        "tfidf_matrix": tfidf_matrix,
        "clusterer": clusterer,
        "labels": labels,
        "topic_model": tm,
        "explorer": ea,
        "ir": ir,
        "recommender": recommender,
        "results": results,
    }


def main():
    run_pipeline(save_cleaned_file=True, visualize=True)


if __name__ == "__main__":
    main() 