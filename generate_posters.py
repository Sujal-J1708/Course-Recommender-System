import pickle
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time
import os

# Load data safely
try:
    df = pickle.load(open("course.pkl", "rb"))
except FileNotFoundError:
    print("❌ course.pkl not found!")
    exit()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

DEFAULT_POSTER = "https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg"


def fetch_poster(url):
    """Fetch course poster with improved error handling and retry logic"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes

        soup = BeautifulSoup(response.text, "html.parser")

        # Try multiple meta tag options
        meta_selectors = [
            soup.find("meta", property="og:image"),
            soup.find("meta", attrs={"name": "og:image"}),
            soup.find("meta", attrs={"name": "twitter:image"})
        ]

        for meta in meta_selectors:
            if meta and meta.get("content"):
                poster_url = meta.get("content")
                # Validate URL format
                if poster_url.startswith(('http://', 'https://')):
                    return poster_url

        # Fallback: look for large images in the page
        images = soup.find_all("img", src=True)
        for img in images:
            src = img.get("src", "")
            if src and any(keyword in src.lower() for keyword in ['course', 'card', 'thumbnail']):
                if src.startswith('//'):
                    return f"https:{src}"
                elif src.startswith('/'):
                    return f"https://www.coursera.org{src}"
                elif src.startswith('http'):
                    return src

    except requests.exceptions.RequestException as e:
        print(f"⚠️ Request failed for {url}: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected error for {url}: {e}")

    return DEFAULT_POSTER


def backup_data(df, filename):
    """Create backup of data"""
    backup_name = f"backup_{int(time.time())}_{filename}"
    pickle.dump(df, open(backup_name, "wb"))
    return backup_name


def main():
    print("🔄 Fetching real posters from Coursera...\n")

    # Create backup
    backup_file = backup_data(df, "course.pkl")
    print(f"📦 Backup created: {backup_file}")

    # Check if poster column exists
    if "poster" not in df.columns:
        df["poster"] = DEFAULT_POSTER

    # Track progress
    updated_count = 0
    posters = []

    for idx, url in enumerate(tqdm(df["url"], desc="Fetching posters")):
        current_poster = df.iloc[idx]["poster"] if idx < len(df) else DEFAULT_POSTER

        # Only fetch if we don't have a valid poster
        if current_poster == DEFAULT_POSTER or pd.isna(current_poster):
            new_poster = fetch_poster(url)
            posters.append(new_poster)
            if new_poster != DEFAULT_POSTER:
                updated_count += 1
            time.sleep(0.5)  # Be respectful to the server
        else:
            posters.append(current_poster)

    df["poster"] = posters

    # Save results
    pickle.dump(df, open("course.pkl", "wb"))

    print(f"\n🎉 Successfully updated {updated_count} posters!")
    print(f"💾 Data saved to course.pkl")


if __name__ == "__main__":
    main()