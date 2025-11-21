import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Smart Course Recommender",
    layout="wide",
    page_icon="🎓"
)


# ---------------------------------------------------
# LOAD REAL COURSERA DATA ONLY
# ---------------------------------------------------
@st.cache_data
def load_data():
    """Load real Coursera data - NO FALLBACK"""
    try:
        df = pickle.load(open("course.pkl", "rb"))
        return df
    except Exception as e:
        st.error(f"❌ Error loading course data: {str(e)}")
        st.stop()  # Stop the app if data fails


# ---------------------------------------------------
# COMPUTE SIMILARITY WITH REAL DATA
# ---------------------------------------------------
@st.cache_data
def compute_similarity(df):
    """Compute similarity matrix using course content"""
    # Combine relevant features for better recommendations
    if 'description' in df.columns:
        features = df['name'] + " " + df['description']
    else:
        features = df['name']

    # Create TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(features)

    # Compute cosine similarity
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity


# ---------------------------------------------------
# RECOMMENDATION FUNCTION
# ---------------------------------------------------
def recommend(course_name, df, similarity, n_recommendations=6):
    """Get course recommendations based on similarity"""
    try:
        if course_name not in df['name'].values:
            return []

        idx = df[df['name'] == course_name].index[0]
        scores = similarity[idx]

        # Get top similar courses (excluding the course itself)
        similar_indices = scores.argsort()[::-1][1:n_recommendations + 1]

        results = []
        for i in similar_indices:
            if i < len(df):
                course_data = {
                    "name": df.iloc[i]["name"],
                    "url": df.iloc[i]["url"],
                    "poster": df.iloc[i]["poster"],
                    "description": df.iloc[i].get("description", "Explore this course!"),
                    "provider": df.iloc[i].get("provider", "Coursera"),
                    "similarity_score": f"{scores[i]:.2f}"
                }
                results.append(course_data)

        return results

    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return []


# ---------------------------------------------------
# STYLING
# ---------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
    }
    .card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        height: 100%;
    }
    .card:hover {
        transform: translateY(-5px);
        border-color: #4A90E2;
        box-shadow: 0 10px 25px rgba(74, 144, 226, 0.2);
    }
    .thumbnail {
        width: 100%;
        height: 180px;
        object-fit: cover;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .course-title {
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 8px;
        color: white;
        line-height: 1.4;
    }
    .course-provider {
        font-size: 14px;
        color: #4A90E2;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .course-desc {
        font-size: 14px;
        color: #94a3b8;
        margin-bottom: 12px;
        line-height: 1.5;
    }
    .similarity-badge {
        background: rgba(74, 144, 226, 0.2);
        color: #4A90E2;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    .btn {
        display: inline-block;
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
        color: white !important;
        padding: 10px 20px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        text-align: center;
        transition: all 0.3s ease;
        width: 100%;
        border: none;
    }
    .btn:hover {
        background: linear-gradient(135deg, #357ABD 0%, #4A90E2 100%);
        transform: translateY(-2px);
    }
    .header {
        background: rgba(255, 255, 255, 0.05);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# MAIN APP
# ---------------------------------------------------
def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1 style="margin:0; color:white;">🎓 Smart Course Recommender</h1>
        <p style="margin:0; color:#94a3b8; font-size:18px;">Course recommendations using real Coursera data</p>
    </div>
    """, unsafe_allow_html=True)

    # Load REAL data only
    df = load_data()
    similarity = compute_similarity(df)

    # Course selection
    st.markdown("---")
    col1, col2 = st.columns([3, 1])

    with col1:
        selected_course = st.selectbox(
            "🔍 **Select a course to get personalized recommendations:**",
            df['name'].unique(),
            help="Choose from 5411 real Coursera courses"
        )

    with col2:
        st.write("<br>", unsafe_allow_html=True)
        recommend_clicked = st.button("✨ **Get Recommendations**", use_container_width=True)

    # Display selected course info
    if selected_course:
        selected_info = df[df['name'] == selected_course].iloc[0]
        st.markdown(f"""
        <div style="background: rgba(74, 144, 226, 0.1); padding: 20px; border-radius: 10px; border-left: 4px solid #4A90E2; margin: 20px 0;">
            <h4 style="margin:0 0 8px 0; color:white;">📚 {selected_info['name']}</h4>
            <p style="margin:0; color:#94a3b8; font-size:14px;">{selected_info.get('description', 'Popular course among learners worldwide')}</p>
        </div>
        """, unsafe_allow_html=True)

    # Recommendations
    if recommend_clicked:
        with st.spinner("🤖 Finding the perfect courses for you..."):
            recommendations = recommend(selected_course, df, similarity, 6)

        if recommendations:
            st.markdown("### 🎯 Recommended For You")
            st.write("Based on your selection, here are courses you might like:")

            # Display recommendations in a grid
            cols = st.columns(3)
            for i, rec in enumerate(recommendations):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class='card'>
                        <img src="{rec['poster']}" class="thumbnail" alt="{rec['name']}" 
                             onerror="this.src='https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg'">
                        <div class="similarity-badge">Match: {rec['similarity_score']}</div>
                        <div class="course-title">{rec['name']}</div>
                        <div class="course-provider">🏛️ {rec['provider']}</div>
                        <div class="course-desc">{rec['description'][:100]}...</div>
                        <a href="{rec['url']}" target="_blank" class="btn">Explore Course</a>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No recommendations found. Please try selecting a different course.")

    # Trending courses section
    st.markdown("---")
    st.markdown("### 🔥 Trending Courses")

    # Get trending courses from real data
    trending_courses = df.sample(min(6, len(df)))

    cols = st.columns(3)
    for i, (_, row) in enumerate(trending_courses.iterrows()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='card'>
                <img src="{row['poster']}" class="thumbnail" alt="{row['name']}" 
                     onerror="this.src='https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg'">
                <div class="course-title">{row['name']}</div>
                <div class="course-provider">🏛️ {row.get('provider', 'Coursera')}</div>
                <div class="course-desc">{row.get('description', 'Popular course among learners')[:100]}...</div>
                <a href="{row['url']}" target="_blank" class="btn">View Course</a>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()