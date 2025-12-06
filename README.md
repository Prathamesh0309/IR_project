## NYC 311 Information Retrieval System
_A Modular, Reproducible Architecture for Text Retrieval, Clustering, and Recommendation on Large-Scale Civic Data._

## Abstract:<br>
This project presents a complete Information Retrieval (IR) pipeline designed for the NYC 311 Service Requests dataset, a large and diverse collection of citizen-reported civic issues. The goal of the system is to transform raw text complaints into a structured, searchable, and interpretable information space. By combining systematic data cleaning, TF–IDF–based vectorization, unsupervised clustering, cosine similarity retrieval, and a lightweight recommendation engine, the project demonstrates how classical IR approaches can be effectively deployed on real-world public service datasets.
In addition to the backend components, the project includes an interactive Streamlit interface that enables users to perform searches, examine cluster groupings, and explore similar complaint patterns. This makes the system useful not only for academic experimentation, but also for exploratory analysis by policy researchers, data analysts, and city-service teams.

## 🏛️ Introduction:<br>
The NYC 311 dataset provides a unique IR challenge because of its scale, heterogeneity, and noise. Complaints vary widely in length, structure, vocabulary, and specificity, making it essential to adopt a rigorous approach to text preprocessing and representation. This project implements a modular architecture that reflects foundational IR concepts while being extendable to more advanced NLP techniques.
The pipeline is intentionally transparent, interpretable, and reproducible. Each stage from cleaning to clustering to retrieval is encapsulated into a dedicated module to support experimentation and analysis. This aligns with academic best practices in IR research, where isolating components enables clearer evaluation and methodological improvements.<br>
**NYC 311 Complaints Dataset:-** https://www.kaggle.com/datasets/new-york-city/ny-311-service-requests
## Dataset:
We use the publicly available NYC 311 Service Requests dataset, which contains citizen-reported issues across New York City. Each entry includes structured fields (borough, coordinates, timestamps) and an unstructured complaint description.
The free-text complaint descriptions are treated as the document corpus for retrieval, making this dataset an excellent candidate for evaluating text cleaning, TF–IDF vectorization, clustering, and similarity-based search.


## 📐System Architecture:<br>
The overall structure of the pipeline reflects classical IR design: text normalization, vector space representation, document clustering, similarity-based retrieval, and downstream recommendation. Below is a diagram that captures this flow at a conceptual level.<br>
<img width="468" height="213" alt="image" src="https://github.com/user-attachments/assets/595a2d57-a77b-40a0-92f1-b86ed34aacec" />

 

**🔍Detailed Component Descriptions:**
1. Data Cleaner
The Data Cleaner is responsible for transforming raw text into a form suitable for vectorization. Real-world datasets like NYC 311 include noise such as HTML fragments, inconsistent casing, emojis, and incomplete sentences. The cleaner applies normalization methods, removes stopwords, and handles missing values, ensuring each document is represented consistently. This stage greatly influences the quality of downstream retrieval, as clean text reduces sparsity and improves term weighting.
2. FeatureEngineer (TF–IDF Vectorization)
The FeatureEngineer translates cleaned complaint text into numerical representations using TF–IDF, a classical weighting scheme widely used in IR research. TF–IDF captures both the importance of words within individual documents and their rarity across the corpus.
The resulting sparse matrix forms the basis for both clustering and retrieval. By storing the vectorizer as a model artifact, the project ensures consistent transformations between training and inference.
3. ClusterModel (KMeans Clustering)
The clustering module discovers emergent topics within complaints without relying on manually labeled data. KMeans is used to group complaints into semantically similar clusters. These clusters can reveal insights such as:
•	common noise issues
•	housing-related problems
•	sanitation and rodent reports
Clustering also supports exploratory IR, enabling users to limit retrieval to specific complaint categories if desired.
4. IRSystem (Cosine Similarity Retrieval)
The IRSystem is the core of this project. Queries entered by the user are vectorized using the same TF–IDF space as the documents. Cosine similarity is then used to measure the closeness between the query vector and complaint vectors.
This classic vector space model is efficient, interpretable, and grounded in decades of IR research. The system returns the top-N most relevant complaints, along with their metadata.
5. Recommender (Nearest Neighbor Search)
Beyond keyword-based search, the system includes a recommendation engine that locates complaints semantically similar to a selected complaint ID. This allows analysts to examine patterns, identify repeated issues, or study related behavior in specific neighborhoods.
6. Evaluation Framework
Evaluation is a fundamental aspect of IR research.
This project includes:
•	binary relevance judgments
•	precision and recall calculations
Evaluating retrieval quality helps diagnose vectorization issues, improve cleaning methods, and compare model variants.
## 🎛 Streamlit Interface
The Streamlit frontend enables real-time interaction with all system components.<br>
Users can:<br>
- search complaints with natural language queries
- browse ranked retrieval outputs
- visualize clusters
- explore similar complaints through the recommendation mode
This interface transforms the IR backend into a usable exploratory analytics tool.
## ▶️ Running the System
1. Install Dependencies:
```bash 
pip install -r requirements.txt
```
2. Run streamlit app:
```bash
streamlit run app.py
```
3. Click on run pipeline button: <br>

<img width="877" height="240" alt="image" src="https://github.com/user-attachments/assets/8dcc738a-0ba8-4131-a333-fb3966d36fdc" /><br>

This will run the pipeline for the first time and store the model outcomes, later the model will fetch data from these stored files for better performance.

## 📊 Discussion & Interpretation:
The results demonstrate that even with classical IR techniques, meaningful retrieval performance can be achieved. TF–IDF remains a strong baseline for structured civic datasets where complaints are short, information-dense, and highly domain-specific.
Clustering reveals clear thematic groups without supervision, highlighting the natural structure of 311 complaint behavior.
Future extensions, such as transformer embeddings, topic modeling, or metadata-aware retrieval, can further enrich the system.

## Future Work: 
Potential academic directions include:
- Using BERT/SBERT embeddings for semantic retrieval
- Combining text with location, date, and complaint type
- Performing time-series drift analysis
- Building a hybrid lexical–semantic retrieval engine
- Deploying as a scalable cloud service


