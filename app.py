import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Course Recommender",
    layout="wide",
    page_icon="🎓"
)


## ---------------------------------------------------
# LOAD DATA WITH ERROR HANDLING
# ---------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pickle.load(open("course.pkl", "rb"))
        return df
    except FileNotFoundError:
        st.error("❌ course.pkl not found! Please run generate_posters.py first.")
        st.stop()

@st.cache_data
def compute_similarity(df):
    """Compute similarity matrix on-the-fly"""
    # Combine relevant features for content-based filtering
    features = df['name'] + " " + df.get('description', '') + " " + df.get('skills', '')

    # Create TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(features)

    # Compute cosine similarity
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity

# Load data
df = load_data()

# Compute similarity matrix
with st.spinner("🔄 Computing course similarities..."):
    similarity = compute_similarity(df)


# ---------------------------------------------------
# DATA VALIDATION & CLEANING
# ---------------------------------------------------
def validate_data(df):
    """Validate and clean the dataset"""
    # Ensure required columns exist
    required_columns = ['name', 'url', 'poster']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"❌ Missing required column: {col}")
            st.stop()

    # Clean poster URLs
    default_poster = "https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg"
    df["poster"] = df["poster"].fillna(default_poster)
    df["poster"] = df["poster"].replace("", default_poster)

    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['name', 'url'])
    if len(df) < initial_count:
        st.sidebar.info(f"Removed {initial_count - len(df)} duplicate courses")

    return df


df = validate_data(df)


# ---------------------------------------------------
# RECOMMENDER FUNCTION
# ---------------------------------------------------
def recommend(course_name, n_recommendations=6):
    """Get course recommendations with improved error handling"""
    try:
        # Find the course index
        course_mask = df["name"] == course_name
        if not course_mask.any():
            st.error(f"❌ Course '{course_name}' not found in dataset")
            return []

        idx = df[course_mask].index[0]

        # Get similarity scores
        if isinstance(similarity, np.ndarray):
            distances = similarity[idx]
        else:
            # Handle sparse matrices or other formats
            distances = similarity[idx].toarray().flatten()

        # Get top similar courses
        similar_indices = distances.argsort()[::-1][1:n_recommendations + 1]

        results = []
        for i in similar_indices:
            if i < len(df):  # Ensure index is within bounds
                results.append({
                    "name": df.iloc[i]["name"],
                    "url": df.iloc[i]["url"],
                    "poster": df.iloc[i]["poster"],
                    "score": distances[i]
                })

        return results

    except Exception as e:
        st.error(f"❌ Error generating recommendations: {e}")
        return []


# ---------------------------------------------------
# CLEAN CSS (REMOVED SEARCH-BOX STYLING)
# ---------------------------------------------------
st.markdown("""
<style>
    .sticky {
        position: sticky;
        top: 0;
        background: rgba(14, 17, 23, 0.9);
        backdrop-filter: blur(10px);
        padding: 1rem 0;
        z-index: 999;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
    }

    .card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        height: 100%;
    }

    .card:hover {
        transform: translateY(-5px);
        border-color: #4A90E2;
        box-shadow: 0 8px 25px rgba(74, 144, 226, 0.15);
    }

    .thumbnail {
        width: 100%;
        height: 160px;
        object-fit: cover;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    .course-title {
        font-weight: 600;
        font-size: 1rem;
        line-height: 1.4;
        margin-bottom: 1rem;
        color: white;
    }

    .btn {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
        text-align: center;
    }

    .btn:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        color: white;
    }

    /* Remove the gray background from selectbox area */
    .stSelectbox {
        background: transparent !important;
    }

    div[data-baseweb="select"] {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    div[data-baseweb="select"]:hover {
        border-color: #4A90E2 !important;
    }

    /* Style the button to match */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }

    .stButton button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.header("🔧 Settings")

    # Number of recommendations
    n_recommendations = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=12,
        value=6,
        step=3
    )

    # Dataset info
    st.header("📊 Dataset Info")
    st.write(f"Total courses: {len(df)}")
    st.write(f"Unique courses: {df['name'].nunique()}")

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("""
<div class='sticky'>
    <h1>🎓 Smart Coursera Recommender</h1>
    <p>Discover personalized course recommendations based on your interests</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# CLEAN SEARCH SECTION (NO GRAY BOX)
# ---------------------------------------------------
col1, col2 = st.columns([3, 1])

with col1:
    course_list = sorted(df["name"].unique().tolist(), key=lambda x: x.lower())
    selected_course = st.selectbox(
        "🔍 Choose a course:",
        course_list,
        index=0,
        help="Select a course to get similar recommendations"
    )

with col2:
    st.write("<br>", unsafe_allow_html=True)  # Vertical alignment
    recommend_clicked = st.button("✨ Get Recommendations", use_container_width=True)

# ---------------------------------------------------
# RECOMMENDATIONS SECTION
# ---------------------------------------------------
if recommend_clicked and selected_course:
    with st.spinner("🔄 Finding the best courses for you..."):
        recommendations = recommend(selected_course, n_recommendations)

    if recommendations:
        st.markdown(f'<div class="section-title">🎯 Recommended for "{selected_course}"</div>', unsafe_allow_html=True)

        # Display in grid
        cols = st.columns(3)
        for i, rec in enumerate(recommendations):
            with cols[i % 3]:
                st.markdown(f"""
                    <div class='card'>
                        <img src="{rec['poster']}" class="thumbnail" 
                             onerror="this.src='https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg'"/>
                        <div class="course-title">{rec['name']}</div>
                        <a href="{rec['url']}" target="_blank" class="btn">Explore Course</a>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No recommendations found. Try selecting a different course.")

# ---------------------------------------------------
# TRENDING COURSES SECTION
# ---------------------------------------------------
st.markdown('<div class="section-title">🔥 Trending Courses</div>', unsafe_allow_html=True)

# Get random trending courses (excluding the currently selected one)
trending_courses = df[df["name"] != selected_course].sample(min(6, len(df))).reset_index(drop=True)

if len(trending_courses) > 0:
    cols = st.columns(3)
    for i, (_, row) in enumerate(trending_courses.iterrows()):
        with cols[i % 3]:
            st.markdown(f"""
                <div class='card'>
                    <img src="{row['poster']}" class="thumbnail" 
                         onerror="this.src='https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg'"/>
                    <div class="course-title">{row['name']}</div>
                    <a href="{row['url']}" target="_blank" class="btn">View Course</a>
                </div>
            """, unsafe_allow_html=True)