import streamlit as st
import pandas as pd
from pathlib import Path

from data_cleaner import DataCleaner
from feature_engineering import FeatureEngineer
from cluster_model import ClusterModel
from retrieval_system import IRSystem
from recommendation import Recommender
from exploratory_analysis import ExploratoryAnalysis
from evaluation import evaluate_with_pseudo_labels
from main import run_pipeline
import joblib
from scipy import sparse
import json
import threading
import time

# Configure page
st.set_page_config(
    page_title="NYC 311 Complaints IR System",
    page_icon="🏙️",
    layout="wide"
)

def format_date(dt):
    '''
    This functions helps in formating dates
    '''
    if not isinstance(dt, str) or dt.strip() == "":
        return "Unknown"
    try:
        # Convert ISO timestamp → pretty date
        d = pd.to_datetime(dt)
        return d.strftime("%b %d, %Y — %I:%M %p")
    except:
        return dt

def format_zip(z):
    '''
    This function helps in formating zip codes
    '''
    if pd.isna(z):
        return "Unknown"
    try:
        return str(int(float(z)))  # removes .0
    except:
        return str(z)
    
# Initialize session state for tracking pipeline status
if 'pipeline_run' not in st.session_state:
    st.session_state.pipeline_run = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'show_error' not in st.session_state:
    st.session_state.show_error = False

# ----------------------------
# Helper functions
@st.cache_data
def load_data_from_pickle(path="data/cleaned_311_data.pkl"):
    """Load cleaned dataframe from pickle file. Returns df or None on error."""
    try:
        return pd.read_pickle(path)
    except Exception as e:
        return None

# ----------------------------
# TF-IDF loading/building functions
def load_tfidf_from_disk():
    """Try to load persisted vectorizer and tfidf matrix from disk. Returns (fe, tfidf_matrix) or (None, None)."""
    vec_path = Path('data') / 'vectorizer.joblib'
    tfidf_path = Path('data') / 'tfidf.npz'
    if vec_path.exists() and tfidf_path.exists():
        try:
            loaded_vec = joblib.load(str(vec_path))
            loaded_tfidf = sparse.load_npz(str(tfidf_path))
            fe_local = FeatureEngineer(max_features=getattr(loaded_vec, 'max_features', 50000))
            fe_local.vectorizer = loaded_vec
            return fe_local, loaded_tfidf
        except Exception as e:
            print(f"Failed to load persisted TF-IDF artifacts: {e}")
            return None, None
    return None, None

# ---------------------------- 
# TF-IDF building and persisting
def build_and_persist_tfidf(df_src, max_features=50000):
    """Build TF-IDF from df_src, persist vectorizer and matrix to disk, return (fe, tfidf_matrix)."""
    try:
        fe_local = FeatureEngineer(max_features=max_features)
        tfidf_matrix_local = fe_local.fit_transform(df_src)
    except Exception as e:
        st.error(f"Error creating TF-IDF: {e}")
        return None, None

    Path('data').mkdir(exist_ok=True)
    # Saves models to local files
    try:
        joblib.dump(fe_local.vectorizer, 'data/vectorizer.joblib')
        sparse.save_npz('data/tfidf.npz', tfidf_matrix_local)
        with open('data/manifest.json', 'w') as mf:
            json.dump({'df_shape': list(getattr(df_src, 'shape', []))}, mf)
        # update session flags
        st.session_state['tfidf_ready'] = True
        st.session_state['vectorizer_path'] = 'data/vectorizer.joblib'
        st.session_state['tfidf_path'] = 'data/tfidf.npz'
    except Exception as e:
        st.warning(f"Could not persist TF-IDF artifacts: {e}")

    return fe_local, tfidf_matrix_local

# ---------------------------- 
# Main TF-IDF getter
def get_or_build_tfidf(df_src=None, max_features=50000):
    """Return (fe, tfidf_matrix). Prefer disk artifacts, else build from df_src.
    Sets session_state flags when successful.
    """
    # Try load from disk first
    fe_loaded, tfidf_loaded = load_tfidf_from_disk()
    if fe_loaded is not None and tfidf_loaded is not None:
        st.session_state['tfidf_ready'] = True
        st.session_state['vectorizer_path'] = 'data/vectorizer.joblib'
        st.session_state['tfidf_path'] = 'data/tfidf.npz'
        return fe_loaded, tfidf_loaded

    # Build from provided df_src
    if df_src is None:
        return None, None
    fe_built, tfidf_built = build_and_persist_tfidf(df_src, max_features=max_features)
    return fe_built, tfidf_built

# ----------------------------
# Background pipeline runner
def _bg_run_pipeline(cleaned_path):
    """Background runner for the pipeline. Writes markers into session_state.
    Runs in a daemon thread so the UI stays responsive. The pipeline itself writes
    persisted artifacts to disk (cleaned pickle, vectorizer, tfidf) which the app
    will use as the canonical source.
    """
    try:
        st.session_state['pipeline_running'] = True
    except Exception:
        pass
    # Run the pipeline
    try:
        run_pipeline(save_cleaned_file=True, cleaned_path=cleaned_path)
        try:
            st.session_state['pipeline_success'] = True
        except Exception:
            pass
    except Exception as e:
        try:
            st.session_state['pipeline_error'] = str(e)
        except Exception:
            pass
    finally:
        try:
            st.session_state['pipeline_running'] = False
        except Exception:
            pass

# ----------------------------
# Start background pipeline
def start_pipeline_background(cleaned_path):
    if st.session_state.get('pipeline_running'):
        return
    t = threading.Thread(target=_bg_run_pipeline, args=(cleaned_path,), daemon=True)
    t.start()
    st.session_state['pipeline_running'] = True

# Main app

# Check for data file
pickle_path = "data/cleaned_311_data.pkl"
df = None
fe = None
tfidf_matrix = None

# Sidebar for data and pipeline controls
with st.sidebar:
    st.header("Data Pipeline Controls")
    
    if not Path(pickle_path).exists():
        st.warning("⚠️ Cleaned data not found")
        run_pipeline_btn = st.button("🚀 Run Full Pipeline")
        if run_pipeline_btn:
            # Start the pipeline in a background thread so the UI doesn't block.
            try:
                Path("data").mkdir(exist_ok=True)
                start_pipeline_background(pickle_path)
                st.success("✅ Pipeline started in background.")
                st.info("It may take several minutes. Use 'Verify artifacts' below to watch progress.")
            except Exception as e:
                st.error("❌ Could not start pipeline in background. See terminal for details.")
                st.session_state.show_error = True
    else:
        st.success("✅ Data file found")
        if st.button("🔄 Re-run Pipeline"):
            try:
                Path("data").mkdir(exist_ok=True)
                start_pipeline_background(pickle_path)
                st.success("✅ Pipeline re-run started in background.")
                st.info("It may take several minutes. Use 'Verify artifacts' below to watch progress.")
            except Exception as e:
                st.error("Error starting pipeline re-run")
                st.session_state.show_error = True


    # ----------------------------
    # Verify artifacts panel
    st.markdown("---")
    st.header("Artifacts & Status")
    vec_path = Path('data') / 'vectorizer.joblib'
    tfidf_path = Path('data') / 'tfidf.npz'
    manifest_path = Path('data') / 'manifest.json'
    cleaned_exists = Path(pickle_path).exists()
    st.write("Cleaned pickle:", str(pickle_path), "—", "✅" if cleaned_exists else "❌")
    st.write("Vectorizer:", str(vec_path), "—", "✅" if vec_path.exists() else "❌")
    st.write("TF-IDF:", str(tfidf_path), "—", "✅" if tfidf_path.exists() else "❌")
    st.write("Manifest:", str(manifest_path), "—", "✅" if manifest_path.exists() else "❌")

    # Show background pipeline status (if any)
    if st.session_state.get('pipeline_running'):
        st.info("Pipeline is running in background...")
    elif st.session_state.get('pipeline_success'):
        st.success("Pipeline completed successfully.")
    elif st.session_state.get('pipeline_error'):
        st.error(f"Pipeline error: {st.session_state.get('pipeline_error')}")

# Load data if not already loaded through pipeline
if df is None and Path(pickle_path).exists():
    # Prefer session_state copy if available (persisted after running pipeline in-app)
    if 'df' in st.session_state and st.session_state.get('df') is not None:
        df = st.session_state.get('df')
        fe = st.session_state.get('fe')
        tfidf_matrix = st.session_state.get('tfidf_matrix')
        st.session_state.data_loaded = True
    else:
        df = load_data_from_pickle(pickle_path)
        if df is not None:
            st.session_state.data_loaded = True

# ----------------------------
# Main content area
st.title("NYC 311 Complaints IR System")

if st.session_state.data_loaded and df is not None:
    st.write("Search past complaints and get recommendations!")
    # User query input
    query = st.text_input("Enter your complaint/query:", "")
    top_n = st.slider("Number of similar complaints to retrieve:", min_value=1, max_value=100, value=5)

    # Prepare features if needed (load from disk or build)
    if fe is None or tfidf_matrix is None:
        fe, tfidf_matrix = get_or_build_tfidf(df)

    if fe is None or tfidf_matrix is None:
        st.error("Feature preparation failed. Please run the pipeline and try again.")
    else:
        # Ensure clusters exist
        if 'cluster' not in df.columns:
            try:
                clusterer = ClusterModel(n_clusters=5)
                labels = clusterer.kmeans(tfidf_matrix)
                df['cluster'] = labels
            except Exception:
                st.warning("Could not compute clusters. Some features may be limited.")

        # Retrieval + Recommendation
        if query:
            try:
                # Ensure the query is preprocessed the same way as training text
                cleaner = DataCleaner()
                clean_q = cleaner.clean_text(query)

                # If TF-IDF artifacts are not in memory, try loading from disk automatically
                if (fe is None or tfidf_matrix is None):
                    vec_path = Path('data') / 'vectorizer.joblib'
                    tfidf_path = Path('data') / 'tfidf.npz'
                    if vec_path.exists() and tfidf_path.exists():
                        try:
                            loaded_vec = joblib.load(str(vec_path))
                            loaded_tfidf = sparse.load_npz(str(tfidf_path))
                            fe = FeatureEngineer()
                            fe.vectorizer = loaded_vec
                            tfidf_matrix = loaded_tfidf
                            st.session_state['tfidf_ready'] = True
                        except Exception as e:
                            st.error('Failed loading TF-IDF artifacts from disk.')
                            with st.expander('Load error'):
                                st.write(str(e))

                ir = IRSystem(tfidf_matrix=tfidf_matrix, df=df)
                results = ir.query(clean_q, vectorizer=fe.vectorizer, top_n=top_n)
                recommender = Recommender()

                # Choose a safe display column (avoid assuming 'Descriptor' exists)
                candidates = ['Descriptor', 'DETAILED_DESCRIPTION', 'BRIEF_DESCRIPTION', 'description', 'cleaned_text']
                display_col = next((c for c in candidates if c in df.columns), None)
                if display_col is None:
                    # fallback to the first column
                    display_col = df.columns[0] if len(df.columns) > 0 else None

                st.subheader("Top Similar Complaints & Recommendations")

                if results is None or results.empty:
                    st.info("No similar complaints found for this query.")
                else:
                    for i, row in results.iterrows():
                        with st.container():
                            col1, col2 = st.columns([3,1])
                            with col1:
                                comp_type = row.get("Complaint Type", "")
                                descriptor = row.get("Descriptor", "")

                                date = format_date(row.get("Created Date", ""))
                                borough = row.get("Borough", "Unknown")
                                zipc = format_zip(row.get("Incident Zip", ""))

                                st.markdown(f"""
                                ### {comp_type} — {descriptor}

                                <span style='color:#aaa;'>
                                📅 <b>{date}</b>  
                                📍 <b>{borough.title()}</b> (ZIP: {zipc})
                                </span>

                                <br>

                                <span style='color:#bbb;'>Similarity:</span>  
                                <b>{row['similarity_score']:.3f}</b>
                                """, unsafe_allow_html=True)
                            with col2:
                                if 'cluster' in df.columns:
                                    recommended_service = recommender.recommend(results)
                                    st.info(f"**Recommended Service:**\n{recommended_service}")
                        st.divider()
            except Exception as e:
                st.error("Error processing your search. Expand for details.")
                with st.expander("Error details"):
                    st.write(str(e))

            # Debugging info to help diagnose empty results
            with st.expander("Debug: pipeline state"):
                st.write({
                    'fe_present': fe is not None,
                    'tfidf_matrix_shape': getattr(tfidf_matrix, 'shape', None),
                    'tfidf_nnz': getattr(tfidf_matrix, 'nnz', None),
                    'vectorizer_vocab_size': len(getattr(fe.vectorizer, 'vocabulary_', {})) if fe is not None else None,
                    'query_raw': query,
                    'query_clean': clean_q,
                })
                if 'results' in locals() and results is not None:
                    try:
                        top_scores = results['similarity_score'].tolist()[:5]
                        st.write('top_scores (head):', top_scores)
                    except Exception:
                        st.write('no similarity scores available')
            run_eval = st.button("Run Evaluation")
            st.subheader("📊 Retrieval System Evaluation")
            # st.write("Columns:", df.columns.tolist())

            try:
                # Ensure TF-IDF and vectorizer are available
                if fe is None or tfidf_matrix is None:
                    fe, tfidf_matrix = get_or_build_tfidf(df)

                # Load IR system
                ir_eval = IRSystem(tfidf_matrix=tfidf_matrix, df=df)

                # Run evaluation
                eval_results = evaluate_with_pseudo_labels(ir_eval, fe, df)

                # Show resulting table
                eval_df = pd.DataFrame(eval_results)
                st.dataframe(eval_df)

            except Exception as e:
                st.error("Evaluation could not be computed.")
                with st.expander("Show Error"):
                    st.write(str(e))
        # Exploratory Analysis toggles
        with st.sidebar:
            st.markdown("---")
            st.header("📊 Exploratory Analysis")
            st.markdown("Advanced exploratory (grouping & seasonal)")
            # Detect possible group columns
            try:
                ea_tmp = ExploratoryAnalysis(df)
                group_candidates = [ea_tmp.detect_group_column()] if ea_tmp.detect_group_column() is not None else []
                # Add a few likely columns if present
                for c in ['Borough','Agency Name', 'Complaint Type', 'full_text', 'cluster']:
                    if c in df.columns and c not in group_candidates:
                        group_candidates.append(c)
            except Exception:
                group_candidates = []
            group_column = st.selectbox("Group by (for top groups / seasonal)", options=[None] + group_candidates, index=0)
            top_n = st.slider("Top N groups to show", min_value=3, max_value=20, value=6)
            show_dashboard = st.button("Show exploratory dashboard")

    # Allow dashboard rendering when user requests it even if the simple toggles are off
    if show_dashboard:  #show_trend
        try:
            ea = ExploratoryAnalysis(df)
            try:
                ea.convert_dates()
            except KeyError as e:
                st.warning(f"Could not find a date column for plotting: {e}")
                raise
            # ensure there is at least one valid date to plot
            date_col = getattr(ea, 'date_column', None)
            if date_col is None or ea.df[date_col].dropna().empty:
                st.warning(f"No valid '{date_col or 'date'}' values available to plot.")
            else:
                if show_dashboard:
                    # Provide diagnostic info in case detection failed
                    with st.expander("Dashboard debug info"):
                        st.write({
                            'detected_date_column': getattr(ea, 'date_column', None),
                            'available_columns_sample': df.columns.tolist()[:30],
                            'selected_group_column': group_column,
                            'top_n': top_n,
                        })
                    try:
                        fig = ea.plot_dashboard(date_column=date_col, group_column=group_column, top_n=top_n)
                        st.subheader("Exploratory Dashboard")
                        st.pyplot(fig)
                        # show top groups table and allow inspecting reasons
                        if group_column is not None:
                            try:
                                gc, counts = ea.top_groups(group_column=group_column, top_n=top_n)
                                st.write(f"Top {len(counts)} groups by '{gc}'")
                                st.dataframe(counts.reset_index().rename(columns={gc: 'group', 0: 'count'}).rename(columns={"index": gc}))
                                sel = st.selectbox("Inspect reasons for which group?", options=list(counts.index))
                                if sel:
                                    reasons = ea.top_reasons_for_group(sel, group_column=group_column)
                                    st.write(f"Top reasons for {sel}")
                                    st.dataframe(reasons.reset_index().rename(columns={0: 'count'}))
                            except Exception as e:
                                st.warning(f"Could not compute top groups table: {e}")
                    except Exception as e:
                        st.error("Could not build exploratory dashboard")
                        with st.expander('Dashboard error'):
                            st.write(str(e))
        except Exception as e:
            st.error("Error generating visualizations")
            with st.expander('Visualization error'):
                st.write(str(e))
                

elif not Path(pickle_path).exists():
    st.info("Please use the sidebar to run the pipeline and generate the required data.")
else:
    st.error("Error loading data. Please try running the pipeline again.")
