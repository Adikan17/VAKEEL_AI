# backend/src/Lawpal_Scraper/scraper/court_scraper.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

HEADERS = {"User-Agent": "LawPalBot/1.0 (+contact@example.com)"}
RATE_LIMIT = 2

def fetch(url):
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    time.sleep(RATE_LIMIT)
    return r.text

def parse_article(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("title").get_text(strip=True)
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n\n".join(paragraphs)
    return {"title": title, "text": text, "source": base_url}

def crawl_index(index_url, limit=5):
    print(f"Crawling {index_url} ...")
    html = fetch(index_url)
    soup = BeautifulSoup(html, "html.parser")

    links = []
    for a in soup.select("a[href]"):
        href = a["href"]
        full_url = urljoin(index_url, href)
        if full_url not in links and len(links) < limit:
            links.append(full_url)

    results = []
    for link in links:
        try:
            art_html = fetch(link)
            results.append((link, parse_article(art_html, index_url)))
        except Exception as e:
            print("Error fetching", link, e)
    return results
