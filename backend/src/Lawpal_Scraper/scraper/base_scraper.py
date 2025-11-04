# backend/src/Lawpal_Scraper/scraper/base_scraper.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

HEADERS = {"User-Agent": "LawPalBot/1.0 (+contact@example.com)"}
RATE_LIMIT = 2  # seconds between requests so you don’t get blocked

def fetch(url):
    """Fetch raw HTML from a URL with rate limiting and error handling."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        time.sleep(RATE_LIMIT)
        return resp.text
    except Exception as e:
        print(f"[fetch error] {url}: {e}")
        return ""

def parse_article(html, base_url=None):
    """Extract title and all paragraph text from an article page."""
    if not html:
        return {"title": "Untitled", "text": "", "source": base_url}

    soup = BeautifulSoup(html, "html.parser")

    title = ""
    if soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)
    elif soup.title:
        title = soup.title.get_text(strip=True)
    else:
        title = "Untitled"

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n\n".join(paragraphs)

    return {"title": title, "text": text, "source": base_url or ""}

def crawl_links(index_url, selector="a[href]", domain_filter=None, limit=10):
    """
    Collect article links from a page using the given CSS selector.
    Example:
        crawl_links("https://example.com/news", "a.article-link", "example.com", 10)
    """
    html = fetch(index_url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a in soup.select(selector):
        href = a.get("href")
        if not href:
            continue
        full_url = urljoin(index_url, href)
        if domain_filter and domain_filter not in full_url:
            continue
        if full_url not in links:
            links.append(full_url)
        if len(links) >= limit:
            break

    return links

def crawl_index(index_url, selector="a[href]", domain_filter=None, limit=5):
    """High-level helper: fetch index page, extract links, parse each article."""
    print(f"Crawling {index_url} ...")
    links = crawl_links(index_url, selector, domain_filter, limit)
    results = []
    for link in links:
        try:
            html = fetch(link)
            article = parse_article(html, base_url=index_url)
            results.append((link, article))
        except Exception as e:
            print(f"[crawl_index error] {link}: {e}")
    return results
