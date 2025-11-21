# 🎓 Smart Coursera Recommender System

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

A intelligent content-based course recommendation system that suggests relevant Coursera courses using machine learning and web scraping.

## 🌐 Live Demo
<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://course-recommender-system-mzldyyvqvrcdcxtdwljshe.streamlit.app/)

**🔗 Direct Link:** https://course-recommender-system-mzldyyvqvrcdcxtdwljshe.streamlit.app/
</div>

## ✨ Features

- **🤖 Smart Recommendations**: Content-based filtering using cosine similarity
- **🖼️ Real Course Posters**: Automated poster fetching from Coursera
- **🎨 Beautiful UI**: Modern Streamlit interface with professional styling
- **⚡ Fast & Lightweight**: Optimized for quick recommendations
- **🔍 Easy Search**: Intuitive course selection and discovery
- **📱 Responsive Design**: Works perfectly on desktop and mobile

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: Scikit-Learn, Pandas, NumPy
- **Web Scraping**: BeautifulSoup, Requests
- **Data Processing**: Pickle, tqdm

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
  git clone https://github.com/Sujal-J1708/Course-Recommender-System.git
   
2. **Install dependencies**
  pip install -r requirements.txt

3. **Run the application**
  streamlit run app.py

## 📁 Project Structure
Course-Recommender-System/

──> app.py                  

──> requirements.txt      

──> LICENSE              

──> DATA_LICENSE.md      

──> README.md            

## 🎯 How It Works
1. Data Processing: Course data is processed and vectorized using TF-IDF
2. Similarity Calculation: Cosine similarity matrix is computed between courses
3. Poster Fetching: Course thumbnails are automatically fetched from Coursera
4. Recommendation Engine: Suggests similar courses based on content similarity
5. Web Interface: Clean UI for course selection and recommendation display

## Download the dataset:
Download from: [Coursera Courses Metadata for Analytics 2025 Dataset on Kaggle](https://www.kaggle.com/datasets/longnguyen3774/coursera-courses-metadata-for-analytics-2025)

## 📊 Dataset
This project uses the Coursera Course Dataset from Kaggle:

Dataset: Coursera Courses Metadata for Analytics 2025
License: CC BY-NC-SA 4.0
Author: Phi Long Nguyen

## License Terms:
✅ Allowed: Personal use, research, educational purposes
✅ Required: Proper attribution to original author
❌ Not Allowed: Commercial use

## 📄 License
**Code License: **
The source code in this repository is licensed under the MIT License - see LICENSE file for details.

**Data License: **
The dataset used in this project is licensed under CC BY-NC-SA 4.0 - see DATA_LICENSE.md for complete terms.

## 🔒 Important Notes:
- This project is for non-commercial, educational purposes only
- Commercial use of this project or dataset is prohibited
- You must provide proper attribution if using or modifying this project

## 👨‍💻 Author
Sujal-J1708
-GitHub: @Sujal-J1708

## 🙏 Acknowledgments
- Phi Long Nguyen for the Coursera Courses Metadata for Analytics 2025 Dataset
- Coursera for course data
- Streamlit for the amazing framework
- Scikit-learn for ML capabilities
