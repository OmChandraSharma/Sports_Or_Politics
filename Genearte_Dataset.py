import requests
from bs4 import BeautifulSoup
import nltk
import pandas as pd
import time

nltk.download("punkt")
nltk.download("punkt_tab")

HEADERS = {"User-Agent": "AcademicCrawler/1.0"}
WIKI_BASE = "https://en.wikipedia.org"
TARGET_SENTENCES = 25000
KEYWORD_LIMIT = 500


# --------------------------
# STEP 1: Generate Keywords
# --------------------------

def extract_category_keywords(category_url, limit=500):
    keywords = set()
    next_page = category_url

    while next_page and len(keywords) < limit:
        print("Collecting keywords from:", next_page)
        response = requests.get(next_page, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.select(".mw-category a"):
            title = link.get_text().strip().lower()
            if len(title) > 3:
                keywords.add(title)

        next_link = soup.find("a", string="next page")
        if next_link:
            next_page = WIKI_BASE + next_link["href"]
        else:
            next_page = None

        time.sleep(0.5)

    return list(keywords)[:limit]


# --------------------------
# STEP 2: Title Filtering
# --------------------------

def is_valid_title(title, keywords):
    title_lower = title.lower()

    if "disambiguation" in title_lower:
        return False

    if ":" in title_lower:
        return False

    return any(keyword in title_lower for keyword in keywords)


# --------------------------
# STEP 3: Fetch Page
# --------------------------

def fetch_page(title):
    url = f"{WIKI_BASE}/wiki/{title}"
    try:
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])

        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/wiki/") and ":" not in href:
                links.append(href.split("/wiki/")[1])

        return text, links
    except:
        return "", []


# --------------------------
# STEP 4: Collect Sentences
# --------------------------

def collect_sentences(seed_pages, keywords, label):
    visited = set()
    queue = seed_pages.copy()
    collected = []

    while queue and len(collected) < TARGET_SENTENCES:
        title = queue.pop(0)

        if title in visited:
            continue

        visited.add(title)

        if not is_valid_title(title, keywords):
            continue

        print("Fetching:", title)

        text, links = fetch_page(title)
        sentences = nltk.sent_tokenize(text)

        for sentence in sentences:
            if len(sentence.split()) > 8:
                collected.append((sentence, label))

            if len(collected) >= TARGET_SENTENCES:
                break

        for link in links:
            if link not in visited and is_valid_title(link, keywords):
                queue.append(link)

        time.sleep(0.3)

    return collected


# --------------------------
# MAIN PIPELINE
# --------------------------

def main():
    print("\nGenerating Sports Keywords...")
    sports_keywords = extract_category_keywords(
        "https://en.wikipedia.org/wiki/Category:Sports",
        KEYWORD_LIMIT
    )

    print("\nGenerating Politics Keywords...")
    politics_keywords = extract_category_keywords(
        "https://en.wikipedia.org/wiki/Category:Politics",
        KEYWORD_LIMIT
    )

    print("\nCollecting SPORTS data...")
    sports_data = collect_sentences(["Sports"], sports_keywords, 0)

    print("\nCollecting POLITICS data...")
    politics_data = collect_sentences(["Politics"], politics_keywords, 1)

    df = pd.DataFrame(sports_data + politics_data,
                      columns=["sentence", "label"])

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv("dataset_title_filtered.csv", index=False)

    print("\nDataset created successfully!")
    print("Sports:", len(sports_data))
    print("Politics:", len(politics_data))
    print("Total:", len(df))


if __name__ == "__main__":
    main()
