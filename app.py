import streamlit as st
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


# ---------------------------------------------------
# SAMPLE DATA - WORKS WITHOUT EXTERNAL FILES
# ---------------------------------------------------
def create_sample_data():
    """Create comprehensive sample data"""
    courses = [
        {
            'name': 'Machine Learning by Andrew Ng',
            'url': 'https://www.coursera.org/learn/machine-learning',
            'poster': 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=400',
            'description': 'Learn machine learning fundamentals and applications'
        },
        {
            'name': 'Python for Everybody',
            'url': 'https://www.coursera.org/specializations/python',
            'poster': 'https://images.unsplash.com/photo-1542831371-29b0f74f9713?w=400',
            'description': 'Beginner-friendly Python programming course'
        },
        {
            'name': 'Data Science Fundamentals',
            'url': 'https://www.coursera.org/specializations/jhu-data-science',
            'poster': 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=400',
            'description': 'Comprehensive data science and analysis course'
        },
        {
            'name': 'Deep Learning Specialization',
            'url': 'https://www.coursera.org/specializations/deep-learning',
            'poster': 'https://images.unsplash.com/photo-1725002327301-06d0a9396e8f?w=400',
            'description': 'Advanced neural networks and deep learning'
        },
        {
            'name': 'Web Development Bootcamp',
            'url': 'https://www.coursera.org/specializations/web-design',
            'poster': 'https://images.unsplash.com/photo-1627398242454-45a1465c2479?w=400',
            'description': 'Full-stack web development course'
        },
        {
            'name': 'Artificial Intelligence A-Z',
            'url': 'https://www.coursera.org/learn/ai',
            'poster': 'https://images.unsplash.com/photo-1677442136019-21780ecad995?w=400',
            'description': 'Learn AI concepts and practical applications'
        },
        {
            'name': 'Cloud Computing Basics',
            'url': 'https://www.coursera.org/learn/cloud-computing',
            'poster': 'https://images.unsplash.com/photo-1544197150-b99a580bb7a8?w=400',
            'description': 'Introduction to cloud services and deployment'
        },
        {
            'name': 'Mobile App Development',
            'url': 'https://www.coursera.org/learn/mobile-app-development',
            'poster': 'https://images.unsplash.com/photo-1512941937669-90a1b58e7e9c?w=400',
            'description': 'Build mobile applications for iOS and Android'
        },
        {
            'name': 'Cybersecurity Fundamentals',
            'url': 'https://www.coursera.org/learn/cybersecurity-basics',
            'poster': 'https://images.unsplash.com/photo-1550751827-4bd374c3f58b?w=400',
            'description': 'Learn essential cybersecurity principles'
        }
    ]
    return pd.DataFrame(courses)


# ---------------------------------------------------
# LOAD DATA WITH PROPER ERROR HANDLING
# ---------------------------------------------------
@st.cache_data
def load_data():
    """Load data with fallback to sample data"""
    try:
        # Try to load existing data
        import pickle
        df = pickle.load(open("course.pkl", "rb"))
        return df
    except FileNotFoundError:
        # Fallback to sample data
        st.info("📝 Using sample course data. For full features, run data preparation locally.")
        return create_sample_data()
    except Exception as e:
        # Any other error, use sample data
        st.warning(f"⚠️ Using sample data: {str(e)}")
        return create_sample_data()


# ---------------------------------------------------
# COMPUTE SIMILARITY
# ---------------------------------------------------
@st.cache_data
def compute_similarity(df):
    """Compute similarity matrix dynamically"""
    # Use course names and descriptions for similarity calculation
    features = df['name'] + " " + df.get('description', '')

    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(features)
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity


# ---------------------------------------------------
# RECOMMENDATION FUNCTION
# ---------------------------------------------------
def recommend(course_name, df, similarity, n_recommendations=6):
    """Get course recommendations"""
    try:
        if course_name not in df['name'].values:
            st.error("Course not found in dataset")
            return []

        idx = df[df['name'] == course_name].index[0]
        distances = similarity[idx]
        similar_indices = distances.argsort()[::-1][1:n_recommendations + 1]

        results = []
        for i in similar_indices:
            if i < len(df):
                results.append({
                    "name": df.iloc[i]["name"],
                    "url": df.iloc[i]["url"],
                    "poster": df.iloc[i]["poster"],
                    "description": df.iloc[i].get("description", "")
                })
        return results
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return []


# ---------------------------------------------------
# STYLING
# ---------------------------------------------------
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
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
        margin-bottom: 10px;
        color: white;
        line-height: 1.4;
    }
    .course-desc {
        font-size: 14px;
        color: #94a3b8;
        margin-bottom: 15px;
        line-height: 1.5;
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
        <p style="margin:0; color:#94a3b8; font-size:18px;">Discover personalized course recommendations based on your interests</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data and compute similarity
    df = load_data()
    similarity = compute_similarity(df)

    # Course selection
    col1, col2 = st.columns([3, 1])

    with col1:
        selected_course = st.selectbox(
            "🔍 **Choose a course to get recommendations:**",
            df['name'].unique(),
            help="Select a course you're interested in"
        )

    with col2:
        st.write("<br>", unsafe_allow_html=True)
        recommend_clicked = st.button("✨ **Get Recommendations**", use_container_width=True)

    # Display selected course info
    if selected_course:
        selected_info = df[df['name'] == selected_course].iloc[0]
        st.markdown(f"""
        <div style="background: rgba(74, 144, 226, 0.1); padding: 20px; border-radius: 10px; border-left: 4px solid #4A90E2; margin: 20px 0;">
            <h4 style="margin:0 0 10px 0; color:white;">Selected: {selected_info['name']}</h4>
            <p style="margin:0; color:#94a3b8;">{selected_info.get('description', 'Popular course among students')}</p>
        </div>
        """, unsafe_allow_html=True)

    # Recommendations
    if recommend_clicked:
        with st.spinner("🔍 Finding the best courses for you..."):
            recommendations = recommend(selected_course, df, similarity, 6)

        if recommendations:
            st.markdown("### 🎯 Recommended Courses")
            st.write("Based on your selection, here are some courses you might like:")

            # Display recommendations in a grid
            cols = st.columns(3)
            for i, rec in enumerate(recommendations):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class='card'>
                        <img src="{rec['poster']}" class="thumbnail" alt="{rec['name']}" 
                             onerror="this.src='https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg'">
                        <div class="course-title">{rec['name']}</div>
                        <div class="course-desc">{rec.get('description', 'Check out this course!')}</div>
                        <a href="{rec['url']}" target="_blank" class="btn">Explore Course</a>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No recommendations found. Please try selecting a different course.")

    # Trending courses section
    st.markdown("---")
    st.markdown("### 🔥 Popular Courses")
    st.write("Check out these trending courses:")

    # Sample trending courses (different from main data)
    trending_data = [
        {
            'name': 'AI For Everyone',
            'url': 'https://www.coursera.org/learn/ai-for-everyone',
            'poster': 'https://images.unsplash.com/photo-1677442136019-21780ecad995?w=400',
            'description': 'AI fundamentals for non-technical learners'
        },
        {
            'name': 'Google UX Design',
            'url': 'https://www.coursera.org/professional-certificates/google-ux-design',
            'poster': 'https://images.unsplash.com/photo-1551650975-87deedd944c3?w=400',
            'description': 'User experience design professional certificate'
        },
        {
            'name': 'IBM Data Science',
            'url': 'https://www.coursera.org/professional-certificates/ibm-data-science',
            'poster': 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=400',
            'description': 'Comprehensive data science professional certificate'
        }
    ]

    trending_df = pd.DataFrame(trending_data)
    cols = st.columns(3)
    for i, (_, row) in enumerate(trending_df.iterrows()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='card'>
                <img src="{row['poster']}" class="thumbnail" alt="{row['name']}" 
                     onerror="this.src='https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg'">
                <div class="course-title">{row['name']}</div>
                <div class="course-desc">{row['description']}</div>
                <a href="{row['url']}" target="_blank" class="btn">View Course</a>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()