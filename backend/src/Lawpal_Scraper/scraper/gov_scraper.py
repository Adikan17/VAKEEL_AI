# scrapers/gov_site_scraper.py
import requests, time, hashlib
from bs4 import BeautifulSoup
from urllib.parse import urljoin

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/118.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

RATE_LIMIT_SECONDS = 2

def fetch_url(url):
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    time.sleep(RATE_LIMIT_SECONDS)
    return r.text

def parse_article(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    # Example: title and all paragraph text inside main content
    title = (soup.find("h1") or soup.title).get_text(strip=True)
    main = soup.find("main") or soup.find("article") or soup
    paragraphs = [p.get_text(" ", strip=True) for p in main.find_all("p")]
    text = "\n\n".join(paragraphs)
    # try to extract publish date
    date = None
    meta_date = soup.find("meta", {"property":"article:published_time"}) or soup.find("time")
    if meta_date:
        date = meta_date.get("datetime") or meta_date.get_text(strip=True)
    return {"title": title, "text": text, "date": date, "source": base_url}

def crawl_index(index_url, limit=20):
    index_html = fetch_url(index_url)
    soup = BeautifulSoup(index_html, "html.parser")
    links = []
    # customize selector per site
    for a in soup.select("a"):
        href = a.get("href")
        if not href: 
            continue
        full = urljoin(index_url, href)
        if "/pdf" in full.lower():  # skip PDFs here (handle separately)
            continue
        # basic domain filter
        if "example.gov" in full:
            links.append(full)
    # unique & limit
    links = list(dict.fromkeys(links))[:limit]
    results = []
    for u in links:
        try:
            html = fetch_url(u)
            results.append((u, parse_article(html, index_url)))
        except Exception as e:
            print("fetch err", u, e)
    return results
