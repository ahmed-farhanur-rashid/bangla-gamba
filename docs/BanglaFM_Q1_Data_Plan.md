# BanglaFM Q1 Data Collection & Augmentation Plan
## Implementation Spec for Scripted Execution

---

## OBJECTIVE

Build a two-contribution paper:
1. **Primary:** First large-scale, documented, Bangladeshi-majority Bangla corpus released as a public artifact
2. **Secondary:** First foundation model with explicit Banglish (romanized Bangla) pretraining

Option 4 (beating BanglaBERT F1 ≥ 72.89 on SentNoB) is not engineered — it either happens from good pretraining or it doesn't. Do not design around it; design around Options 1 and 3 and report Option 4 honestly.

**Target venue:** ACM TALLIP (Q2, Transactions on Asian and Low-Resource Language Information Processing) or LREC-COLING 2026  
**Stretch target:** EMNLP Findings 2026  
**What makes it Q1-adjacent:** Novel corpus artifact + novel Banglish pretraining + reproducible evaluation against established baselines

---

## EVALUATION CRITERIA

### Primary Paper Metric
**Macro-F1 on SentNoB** (3-class: Positive / Negative / Neutral)
- BanglaBERT baseline: 72.89 — this is the number to beat
- Why macro-F1: robust to class imbalance, directly comparable across all prior Bangla NLP work
- Dataset: https://github.com/csebuetnlp/SentNoB

### Secondary Metrics
**BLUB Benchmark** (multi-task, harder to dismiss as cherry-picked)
- NLI accuracy
- QA exact match + F1
- NER macro-F1
- Link: https://github.com/csebuetnlp/BLUB

**BanglaGLUE** (6 classification tasks)
- Link: https://github.com/csebuetnlp/banglabert

### Corpus Quality Metrics (Dataset Contribution)
- **Perplexity:** KenLM 5-gram trained on your corpus vs CulturaX bn on same held-out Bangla text — lower = better quality
- **Fertility score:** Average tokens per word on Bangla text using your tokenizer vs GPT-2 tokenizer — lower = better Bangla tokenization
- **Domain coverage:** Number of unique registered domains, formal vs informal source ratio
- **Deduplication rate:** % documents removed — demonstrates corpus hygiene rigor
- **BD:WB ratio:** Estimated from source metadata — your primary differentiator from CulturaX/Sangraha

### NOT BLEU
BLEU measures n-gram overlap against translation references. It is a machine translation metric. Your model is not a translation model. Do not report BLEU as a primary metric.

---

## PART 1: CORPUS SCRAPING

### 1.1 Target Sources and Priority

| Priority | Source | Type | Est. Tokens | BD/WB | Method |
|---|---|---|---|---|---|
| P0 | Prothom Alo | Formal news | 500M–1B | 100% BD | Sitemap scrape |
| P0 | BDNews24 | Formal news | 200M–400M | 100% BD | Sitemap scrape |
| P0 | Kaler Kantho | Formal news | 150M–300M | 100% BD | Sitemap scrape |
| P0 | Daily Ittefaq | Formal news | 100M–200M | 100% BD | Sitemap scrape |
| P1 | Somewhereinblog | Informal blog | 100M–200M | ~95% BD | Pagination scrape |
| P1 | Sachalayatan | Informal blog | 50M–100M | ~95% BD | Pagination scrape |
| P1 | Rokomari reviews | Informal commerce | 30M–60M | 100% BD | API/pagination |
| P1 | Bikroy listings | Informal commerce | 20M–40M | 100% BD | Pagination scrape |
| P2 | Bangladesh Parliament | Formal official | 20M–50M | 100% BD | PDF scrape |
| P2 | NCTB textbooks | Formal education | 15M | 100% BD | Already available |
| P2 | Banglapedia | Formal encyclopedic | 10M–20M | 100% BD | Pagination scrape |

**Total estimate:** 1.2B–2.4B tokens of new, documented, Bangladeshi-majority Bangla not in any existing HuggingFace dataset.

### 1.2 Legal and Ethical Considerations

Before scraping, check robots.txt for each domain. Respect crawl-delay directives. For the paper:
- Cite the robots.txt compliance explicitly in your data section
- Do not redistribute raw scraped HTML — only release cleaned text + metadata
- Include a datasheet (Gebru et al., 2018 format) documenting provenance, collection date, and filtering
- For Prothom Alo specifically: their terms prohibit commercial redistribution. Release the corpus for research use only under CC BY-NC 4.0

### 1.3 Scraping Architecture

Each scraper follows the same pipeline:

```
Sitemap/Pagination Discovery
        ↓
URL List Generation (deduplicated)
        ↓
Trafilatura Extraction (article text + metadata)
        ↓
Raw JSONL Storage (url, date, title, body, domain)
        ↓
Langid Filtering (fasttext, confidence > 0.80)
        ↓
Quality Filter (length, punctuation density)
        ↓
Cleaned JSONL Output
```

**Core library:** `trafilatura` — handles boilerplate removal, navigation extraction, ad stripping automatically. One function call per URL.

```python
import trafilatura

def extract_article(url: str) -> dict | None:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    result = trafilatura.extract(
        downloaded,
        output_format='json',
        include_metadata=True,
        include_comments=False,   # exclude comment sections
        no_fallback=False,
        favor_precision=True,     # prefer precision over recall for quality
    )
    if not result:
        return None
    import json
    return json.loads(result)
```

Install: `pip install trafilatura fasttext requests lxml`

### 1.4 Prothom Alo Scraper

Prothom Alo (prothomalo.com) is your highest-value source — decades of archives, consistent formal Bangladeshi Bangla, structured sitemap.

```python
"""
prothomalo_scraper.py
Scrapes Prothom Alo articles via sitemap index.
Outputs: prothomalo_raw.jsonl (one JSON object per line)
"""

import requests
import time
import json
import gzip
from pathlib import Path
from xml.etree import ElementTree as ET
import trafilatura
from urllib.parse import urlparse

SITEMAP_INDEX = "https://www.prothomalo.com/sitemap_index.xml"
OUTPUT_FILE = Path("data/raw/prothomalo_raw.jsonl")
CRAWL_DELAY = 1.5   # seconds between requests — be polite
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BanglaFM-Research/1.0; +https://github.com/ahmed-farhanur-rashid/banglaFM)"
}

def fetch_sitemap_urls(sitemap_index_url: str) -> list[str]:
    """Fetch all article URLs from sitemap index."""
    resp = requests.get(sitemap_index_url, headers=HEADERS, timeout=30)
    root = ET.fromstring(resp.content)
    ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    
    article_urls = []
    sitemap_locs = [loc.text for loc in root.findall('sm:sitemap/sm:loc', ns)]
    
    for sitemap_loc in sitemap_locs:
        time.sleep(CRAWL_DELAY)
        try:
            if sitemap_loc.endswith('.gz'):
                r = requests.get(sitemap_loc, headers=HEADERS, timeout=30)
                content = gzip.decompress(r.content)
            else:
                r = requests.get(sitemap_loc, headers=HEADERS, timeout=30)
                content = r.content
            
            sub_root = ET.fromstring(content)
            urls = [loc.text for loc in sub_root.findall('sm:url/sm:loc', ns)]
            article_urls.extend(urls)
            print(f"  Fetched {len(urls)} URLs from {sitemap_loc}")
        except Exception as e:
            print(f"  ERROR on {sitemap_loc}: {e}")
    
    return article_urls


def scrape_article(url: str) -> dict | None:
    """Fetch and extract one article. Returns None on failure."""
    try:
        time.sleep(CRAWL_DELAY)
        downloaded = trafilatura.fetch_url(url, decode=True)
        if not downloaded:
            return None
        
        result = trafilatura.extract(
            downloaded,
            output_format='json',
            include_metadata=True,
            include_comments=False,
            favor_precision=True,
            target_language='bn',
        )
        if not result:
            return None
        
        data = json.loads(result)
        # Enforce minimum quality
        text = data.get('text', '')
        if len(text.split()) < 30:   # discard very short articles
            return None
        
        return {
            'url': url,
            'domain': 'prothomalo.com',
            'source_type': 'formal_news',
            'language_region': 'BD',
            'title': data.get('title', ''),
            'date': data.get('date', ''),
            'text': text,
            'word_count': len(text.split()),
        }
    except Exception as e:
        print(f"  ERROR scraping {url}: {e}")
        return None


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Load already-scraped URLs to support resumption
    scraped_urls = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, 'r') as f:
            for line in f:
                try:
                    scraped_urls.add(json.loads(line)['url'])
                except:
                    pass
    print(f"Already scraped: {len(scraped_urls)} articles")
    
    print("Fetching sitemap URLs...")
    all_urls = fetch_sitemap_urls(SITEMAP_INDEX)
    pending = [u for u in all_urls if u not in scraped_urls]
    print(f"Total URLs: {len(all_urls)} | Pending: {len(pending)}")
    
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f:
        for i, url in enumerate(pending):
            article = scrape_article(url)
            if article:
                out_f.write(json.dumps(article, ensure_ascii=False) + '\n')
            
            if (i + 1) % 500 == 0:
                out_f.flush()
                print(f"Progress: {i+1}/{len(pending)} | Written: {i+1}")


if __name__ == '__main__':
    main()
```

**Expected runtime:** 500K–1M articles × 1.5s delay = 8–16 days of continuous scraping. Run in a tmux session. Use a VPS or run overnight in segments. Alternatively, reduce CRAWL_DELAY to 0.5s and add retry logic — monitor for 429 responses.

### 1.5 Generic News Scraper (BDNews24, Kaler Kantho, Ittefaq)

These sites follow the same sitemap pattern. Use a config-driven version of the same scraper:

```python
"""
generic_news_scraper.py
Config-driven scraper for BD news sites with sitemaps.
"""

import requests, time, json, gzip
from pathlib import Path
from xml.etree import ElementTree as ET
import trafilatura

SITE_CONFIGS = {
    'bdnews24': {
        'sitemap_index': 'https://bangla.bdnews24.com/sitemap_index.xml',
        'output': 'data/raw/bdnews24_raw.jsonl',
        'domain': 'bangla.bdnews24.com',
        'language_region': 'BD',
        'source_type': 'formal_news',
        'crawl_delay': 1.0,
    },
    'kalerkantho': {
        'sitemap_index': 'https://www.kalerkantho.com/sitemap_index.xml',
        'output': 'data/raw/kalerkantho_raw.jsonl',
        'domain': 'kalerkantho.com',
        'language_region': 'BD',
        'source_type': 'formal_news',
        'crawl_delay': 1.2,
    },
    'ittefaq': {
        'sitemap_index': 'https://www.ittefaq.com.bd/sitemap_index.xml',
        'output': 'data/raw/ittefaq_raw.jsonl',
        'domain': 'ittefaq.com.bd',
        'language_region': 'BD',
        'source_type': 'formal_news',
        'crawl_delay': 1.0,
    },
    'samakal': {
        'sitemap_index': 'https://samakal.com/sitemap_index.xml',
        'output': 'data/raw/samakal_raw.jsonl',
        'domain': 'samakal.com',
        'language_region': 'BD',
        'source_type': 'formal_news',
        'crawl_delay': 1.0,
    },
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BanglaFM-Research/1.0; +https://github.com/ahmed-farhanur-rashid/banglaFM)"
}

def scrape_site(config: dict):
    output_path = Path(config['output'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Resume support
    scraped = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try: scraped.add(json.loads(line)['url'])
                except: pass
    
    # Fetch all URLs from sitemap
    urls = []
    try:
        resp = requests.get(config['sitemap_index'], headers=HEADERS, timeout=30)
        root = ET.fromstring(resp.content)
        ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        sub_sitemaps = [l.text for l in root.findall('sm:sitemap/sm:loc', ns)]
        
        for sub in sub_sitemaps:
            time.sleep(0.5)
            try:
                r = requests.get(sub, headers=HEADERS, timeout=30)
                content = gzip.decompress(r.content) if sub.endswith('.gz') else r.content
                sub_root = ET.fromstring(content)
                urls += [l.text for l in sub_root.findall('sm:url/sm:loc', ns)]
            except Exception as e:
                print(f"Sitemap error {sub}: {e}")
    except Exception as e:
        print(f"Index error: {e}")
        return
    
    pending = [u for u in urls if u not in scraped]
    print(f"{config['domain']}: {len(pending)} pending")
    
    with open(output_path, 'a', encoding='utf-8') as f:
        for i, url in enumerate(pending):
            time.sleep(config['crawl_delay'])
            try:
                downloaded = trafilatura.fetch_url(url)
                if not downloaded: continue
                result = trafilatura.extract(downloaded, output_format='json',
                    include_metadata=True, include_comments=False, favor_precision=True)
                if not result: continue
                data = json.loads(result)
                text = data.get('text', '')
                if len(text.split()) < 30: continue
                
                record = {
                    'url': url,
                    'domain': config['domain'],
                    'source_type': config['source_type'],
                    'language_region': config['language_region'],
                    'title': data.get('title', ''),
                    'date': data.get('date', ''),
                    'text': text,
                    'word_count': len(text.split()),
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                if (i+1) % 500 == 0: f.flush()
            except Exception as e:
                print(f"Error {url}: {e}")


if __name__ == '__main__':
    import sys
    site = sys.argv[1] if len(sys.argv) > 1 else None
    configs = {k: v for k, v in SITE_CONFIGS.items() if k == site} if site else SITE_CONFIGS
    for name, cfg in configs.items():
        print(f"\n=== Scraping {name} ===")
        scrape_site(cfg)
```

Usage: `python generic_news_scraper.py bdnews24` or run all: `python generic_news_scraper.py`

### 1.6 Blog Scraper (Somewhereinblog, Sachalayatan)

These are pagination-based, not sitemap-based. Different discovery strategy:

```python
"""
blog_scraper.py
Pagination-based scraper for Bangladeshi blogging platforms.
Somewhereinblog: informal, authentic BD colloquial Bangla
Sachalayatan: informal, literary/political BD Bangla
"""

import requests, time, json, re
from pathlib import Path
from bs4 import BeautifulSoup
import trafilatura

BLOG_CONFIGS = {
    'somewhereinblog': {
        # Somewhereinblog has a chronological archive
        'base_url': 'https://www.somewhereinblog.net',
        'archive_url': 'https://www.somewhereinblog.net/blog/all/{page}',
        'start_page': 1,
        'max_pages': 5000,   # adjust based on actual site depth
        'output': 'data/raw/somewhereinblog_raw.jsonl',
        'domain': 'somewhereinblog.net',
        'source_type': 'informal_blog',
        'language_region': 'BD',
        'crawl_delay': 2.0,  # be extra polite on blogs
    },
    'sachalayatan': {
        'base_url': 'https://www.sachalayatan.com',
        'archive_url': 'https://www.sachalayatan.com/blog?page={page}',
        'start_page': 0,
        'max_pages': 3000,
        'output': 'data/raw/sachalayatan_raw.jsonl',
        'domain': 'sachalayatan.com',
        'source_type': 'informal_blog',
        'language_region': 'BD',
        'crawl_delay': 2.0,
    },
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BanglaFM-Research/1.0; +https://github.com/ahmed-farhanur-rashid/banglaFM)"
}

def discover_article_urls(archive_url: str, base_url: str, session: requests.Session) -> list[str]:
    """Extract article links from one archive/listing page."""
    try:
        resp = session.get(archive_url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(resp.content, 'lxml')
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            # Heuristic: article URLs typically have /post/ or /entry/ or are longer paths
            if re.search(r'/(post|entry|blog|article)/\d+', href):
                full = href if href.startswith('http') else base_url + href
                links.append(full)
        return list(set(links))
    except Exception as e:
        print(f"Discovery error {archive_url}: {e}")
        return []


def scrape_blog_site(config: dict):
    output_path = Path(config['output'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    scraped = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try: scraped.add(json.loads(line)['url'])
                except: pass
    
    session = requests.Session()
    
    with open(output_path, 'a', encoding='utf-8') as out_f:
        for page in range(config['start_page'], config['max_pages']):
            archive_url = config['archive_url'].format(page=page)
            time.sleep(config['crawl_delay'])
            
            article_urls = discover_article_urls(archive_url, config['base_url'], session)
            if not article_urls:
                print(f"Page {page}: no URLs found — may have reached end")
                # Don't break immediately — some pages may be empty
                continue
            
            for url in article_urls:
                if url in scraped: continue
                time.sleep(config['crawl_delay'])
                
                try:
                    downloaded = trafilatura.fetch_url(url)
                    if not downloaded: continue
                    result = trafilatura.extract(downloaded, output_format='json',
                        include_metadata=True, include_comments=True,  # include comments for informal register
                        favor_recall=True)   # blogs: prefer recall over precision
                    if not result: continue
                    data = json.loads(result)
                    text = data.get('text', '')
                    if len(text.split()) < 20: continue
                    
                    record = {
                        'url': url,
                        'domain': config['domain'],
                        'source_type': config['source_type'],
                        'language_region': config['language_region'],
                        'title': data.get('title', ''),
                        'date': data.get('date', ''),
                        'text': text,
                        'word_count': len(text.split()),
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    scraped.add(url)
                except Exception as e:
                    print(f"Error {url}: {e}")
            
            if page % 100 == 0:
                out_f.flush()
                print(f"Page {page} done | Total scraped: {len(scraped)}")


if __name__ == '__main__':
    import sys
    site = sys.argv[1] if len(sys.argv) > 1 else None
    configs = {k: v for k, v in BLOG_CONFIGS.items() if k == site} if site else BLOG_CONFIGS
    for name, cfg in configs.items():
        print(f"\n=== Scraping {name} ===")
        scrape_blog_site(cfg)
```

Install: `pip install beautifulsoup4 lxml trafilatura requests`

### 1.7 E-commerce Review Scraper (Rokomari)

Rokomari is Bangladesh's largest book/product platform. Reviews are informal, authentic Bangladeshi colloquial Bangla — the hardest register to find in existing datasets.

```python
"""
rokomari_scraper.py
Scrapes product reviews from Rokomari.com
Reviews are informal, authentic Bangladeshi colloquial Bangla.
Strategy: iterate product categories → product listings → reviews per product
"""

import requests, time, json
from pathlib import Path
from bs4 import BeautifulSoup

OUTPUT = Path("data/raw/rokomari_reviews_raw.jsonl")
BASE_URL = "https://www.rokomari.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BanglaFM-Research/1.0; +https://github.com/ahmed-farhanur-rashid/banglaFM)",
    "Accept-Language": "bn-BD,bn;q=0.9",
}
CRAWL_DELAY = 1.5

CATEGORY_URLS = [
    "/book/category/1",    # Bengali literature
    "/book/category/2",    # Islamic books
    "/book/category/8",    # Children
    "/book/category/9",    # Religion
    "/book/category/11",   # Academic
    "/book/category/16",   # Science
    "/book/category/22",   # History
    # Add more categories as needed
]


def get_product_urls_from_category(category_path: str, session: requests.Session, max_pages: int = 50) -> list[str]:
    product_urls = []
    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}{category_path}?page={page}"
        try:
            time.sleep(CRAWL_DELAY)
            resp = session.get(url, headers=HEADERS, timeout=20)
            soup = BeautifulSoup(resp.content, 'lxml')
            product_links = soup.select('a.details-book-info')
            if not product_links:
                break
            for link in product_links:
                href = link.get('href', '')
                if href:
                    full = href if href.startswith('http') else BASE_URL + href
                    product_urls.append(full)
        except Exception as e:
            print(f"Category page error {url}: {e}")
    return product_urls


def get_reviews_for_product(product_url: str, session: requests.Session) -> list[dict]:
    reviews = []
    page = 1
    while True:
        review_url = f"{product_url}?tab=review&page={page}"
        try:
            time.sleep(CRAWL_DELAY)
            resp = session.get(review_url, headers=HEADERS, timeout=20)
            soup = BeautifulSoup(resp.content, 'lxml')
            review_blocks = soup.select('.review-list__single-review')
            if not review_blocks:
                break
            
            for block in review_blocks:
                text_el = block.select_one('.review-list__description')
                rating_el = block.select_one('.rating-star--fill')
                if not text_el: continue
                text = text_el.get_text(strip=True)
                if len(text.split()) < 5: continue
                
                reviews.append({
                    'url': product_url,
                    'domain': 'rokomari.com',
                    'source_type': 'informal_commerce_review',
                    'language_region': 'BD',
                    'text': text,
                    'rating': len(soup.select('.rating-star--fill')),
                    'word_count': len(text.split()),
                })
            page += 1
        except Exception as e:
            print(f"Review page error {review_url}: {e}")
            break
    return reviews


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    scraped_urls = set()
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            for line in f:
                try: scraped_urls.add(json.loads(line)['url'])
                except: pass
    
    session = requests.Session()
    
    with open(OUTPUT, 'a', encoding='utf-8') as out_f:
        for cat in CATEGORY_URLS:
            print(f"\nCategory: {cat}")
            product_urls = get_product_urls_from_category(cat, session)
            print(f"  Found {len(product_urls)} products")
            
            for prod_url in product_urls:
                reviews = get_reviews_for_product(prod_url, session)
                for review in reviews:
                    out_f.write(json.dumps(review, ensure_ascii=False) + '\n')
                if reviews:
                    out_f.flush()


if __name__ == '__main__':
    main()
```

### 1.8 Bangladesh Parliament Scraper

Parliament proceedings (Jatiyo Sangsad) are 100% formal official Bangladeshi Bangla — the most authoritative register. Available at parliament.gov.bd as PDFs.

```python
"""
parliament_scraper.py
Downloads and extracts text from Bangladesh Parliament (Jatiyo Sangsad)
proceeding PDFs. These are official government documents in formal BD Bangla.
"""

import requests, time, json
from pathlib import Path
from bs4 import BeautifulSoup
import pdfplumber   # pip install pdfplumber

PARLIAMENT_BASE = "https://www.parliament.gov.bd"
PROCEEDINGS_URL = "https://www.parliament.gov.bd/index.php/en/parliamentary-business/proceedings-of-the-parliament"
OUTPUT = Path("data/raw/parliament_raw.jsonl")
PDF_DIR = Path("data/raw/parliament_pdfs")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; BanglaFM-Research/1.0)"}
CRAWL_DELAY = 2.0


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using pdfplumber."""
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        print(f"PDF extract error {pdf_path}: {e}")
    return '\n'.join(text_parts)


def discover_pdf_links(session: requests.Session) -> list[dict]:
    """Crawl parliament site to find proceeding PDF links."""
    pdf_links = []
    try:
        resp = session.get(PROCEEDINGS_URL, headers=HEADERS, timeout=30)
        soup = BeautifulSoup(resp.content, 'lxml')
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.lower().endswith('.pdf'):
                full = href if href.startswith('http') else PARLIAMENT_BASE + href
                pdf_links.append({
                    'url': full,
                    'title': a.get_text(strip=True),
                })
    except Exception as e:
        print(f"Discovery error: {e}")
    return pdf_links


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    
    session = requests.Session()
    pdf_links = discover_pdf_links(session)
    print(f"Found {len(pdf_links)} PDFs")
    
    processed = set()
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            for line in f:
                try: processed.add(json.loads(line)['url'])
                except: pass
    
    with open(OUTPUT, 'a', encoding='utf-8') as out_f:
        for item in pdf_links:
            if item['url'] in processed: continue
            time.sleep(CRAWL_DELAY)
            
            pdf_filename = PDF_DIR / item['url'].split('/')[-1]
            try:
                # Download PDF
                r = session.get(item['url'], headers=HEADERS, timeout=60, stream=True)
                with open(pdf_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Extract text
                text = extract_pdf_text(pdf_filename)
                if len(text.split()) < 50: continue
                
                record = {
                    'url': item['url'],
                    'domain': 'parliament.gov.bd',
                    'source_type': 'formal_official',
                    'language_region': 'BD',
                    'title': item['title'],
                    'text': text,
                    'word_count': len(text.split()),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                processed.add(item['url'])
                out_f.flush()
                print(f"Processed: {item['title'][:60]}")
            except Exception as e:
                print(f"Error {item['url']}: {e}")


if __name__ == '__main__':
    main()
```

Install: `pip install pdfplumber`

---

## PART 2: DATA CLEANING PIPELINE

### 2.1 Full Cleaning Pipeline

Apply this to all scraped JSONL files before combining with existing HuggingFace datasets.

```python
"""
clean_pipeline.py
Full cleaning pipeline for scraped Bangla corpus.
Stages: langid → unicode_norm → quality_filter → dedup → output
Input:  data/raw/*.jsonl
Output: data/cleaned/corpus_cleaned.jsonl
"""

import json, unicodedata, re, hashlib
from pathlib import Path
from collections import defaultdict

# pip install fasttext bnunicodenormalizer datasketch langdetect
import fasttext
from bnunicodenormalizer import Normalizer
from datasketch import MinHash, MinHashLSH

# ── Config ────────────────────────────────────────────────────────────────────
LANGID_MODEL = "lid.176.bin"          # download from fasttext site
LANGID_THRESHOLD = 0.80               # minimum Bangla confidence
MIN_WORDS = 30                        # minimum document length
MAX_PUNCT_RATIO = 0.30                # max punctuation density
MINHASH_THRESHOLD = 0.80             # Jaccard similarity for dedup
MINHASH_NUM_PERM = 128               # hash permutations for MinHash
NGRAM_SIZE = 5                       # 5-gram shingles for MinHash

RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/cleaned/corpus_cleaned.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Normalizer ────────────────────────────────────────────────────────────────
bnorm = Normalizer()
langid_model = fasttext.load_model(LANGID_MODEL)


def normalize_text(text: str) -> str:
    """Bangla-specific + NFC Unicode normalization."""
    text = bnorm.normalize(text)['normalized']
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_bangla(text: str, threshold: float = LANGID_THRESHOLD) -> bool:
    """Check if text is predominantly Bangla using fasttext langid."""
    # fasttext expects single-line input
    clean = text.replace('\n', ' ')[:500]
    labels, scores = langid_model.predict(clean, k=1)
    label = labels[0].replace('__label__', '')
    return label == 'bn' and scores[0] >= threshold


def quality_filter(text: str) -> bool:
    """Return True if text passes quality checks."""
    words = text.split()
    if len(words) < MIN_WORDS:
        return False
    punct_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if punct_count / max(len(text), 1) > MAX_PUNCT_RATIO:
        return False
    # Reject if more than 30% ASCII (likely English-dominant)
    ascii_count = sum(1 for c in text if ord(c) < 128)
    if ascii_count / max(len(text), 1) > 0.40:
        return False
    return True


def get_minhash(text: str, n: int = NGRAM_SIZE) -> MinHash:
    """Compute MinHash from n-gram shingles."""
    m = MinHash(num_perm=MINHASH_NUM_PERM)
    words = text.split()
    for i in range(len(words) - n + 1):
        shingle = ' '.join(words[i:i+n])
        m.update(shingle.encode('utf-8'))
    return m


def run_pipeline():
    raw_files = sorted(RAW_DIR.glob("*.jsonl"))
    print(f"Found {len(raw_files)} raw files")
    
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_NUM_PERM)
    
    stats = defaultdict(int)
    doc_id = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for raw_file in raw_files:
            print(f"\nProcessing: {raw_file.name}")
            file_stats = defaultdict(int)
            
            with open(raw_file, encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                    except:
                        continue
                    
                    file_stats['total'] += 1
                    text = record.get('text', '')
                    if not text:
                        file_stats['empty'] += 1
                        continue
                    
                    # Stage 1: Language ID
                    if not is_bangla(text):
                        file_stats['langid_reject'] += 1
                        continue
                    
                    # Stage 2: Unicode normalization
                    text = normalize_text(text)
                    record['text'] = text
                    
                    # Stage 3: Quality filter
                    if not quality_filter(text):
                        file_stats['quality_reject'] += 1
                        continue
                    
                    # Stage 4: MinHash deduplication
                    mh = get_minhash(text)
                    key = f"doc_{doc_id}"
                    try:
                        result = lsh.query(mh)
                        if result:
                            file_stats['dedup_reject'] += 1
                            continue
                        lsh.insert(key, mh)
                    except Exception:
                        pass
                    
                    # Stage 5: Write clean record
                    record['doc_id'] = doc_id
                    record['text'] = text
                    out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    doc_id += 1
                    file_stats['kept'] += 1
            
            print(f"  {raw_file.name}: {dict(file_stats)}")
            for k, v in file_stats.items():
                stats[k] += v
    
    print(f"\n=== FINAL STATS ===")
    print(f"Total processed: {stats['total']}")
    print(f"Kept: {stats['kept']} ({100*stats['kept']/max(stats['total'],1):.1f}%)")
    print(f"Rejected by langid: {stats['langid_reject']}")
    print(f"Rejected by quality: {stats['quality_reject']}")
    print(f"Rejected by dedup: {stats['dedup_reject']}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == '__main__':
    run_pipeline()
```

### 2.2 Contamination Check

Remove any document containing verbatim passages from your evaluation benchmarks.

```python
"""
contamination_check.py
Remove documents that contain verbatim n-gram overlap with evaluation benchmarks.
Run AFTER the main cleaning pipeline.
"""

import json, re
from pathlib import Path

# Load evaluation texts (sentences from SentNoB and BLUB)
EVAL_FILES = [
    "eval/sentnob_test.txt",    # one sentence per line
    "eval/blub_test.txt",
]
NGRAM_SIZE = 13  # 13-gram overlap is considered contamination
INPUT = Path("data/cleaned/corpus_cleaned.jsonl")
OUTPUT = Path("data/cleaned/corpus_decontaminated.jsonl")


def load_eval_ngrams(eval_files: list[str], n: int) -> set[str]:
    ngrams = set()
    for ef in eval_files:
        try:
            with open(ef) as f:
                for line in f:
                    words = line.strip().split()
                    for i in range(len(words) - n + 1):
                        ngrams.add(' '.join(words[i:i+n]))
        except FileNotFoundError:
            print(f"Warning: eval file not found: {ef}")
    print(f"Loaded {len(ngrams)} eval n-grams")
    return ngrams


def is_contaminated(text: str, eval_ngrams: set[str], n: int) -> bool:
    words = text.split()
    for i in range(len(words) - n + 1):
        candidate = ' '.join(words[i:i+n])
        if candidate in eval_ngrams:
            return True
    return False


def main():
    eval_ngrams = load_eval_ngrams(EVAL_FILES, NGRAM_SIZE)
    
    kept, removed = 0, 0
    with open(INPUT) as in_f, open(OUTPUT, 'w', encoding='utf-8') as out_f:
        for line in in_f:
            try:
                record = json.loads(line)
                if is_contaminated(record['text'], eval_ngrams, NGRAM_SIZE):
                    removed += 1
                else:
                    out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    kept += 1
            except:
                continue
    
    print(f"Kept: {kept} | Contaminated removed: {removed}")


if __name__ == '__main__':
    main()
```

### 2.3 Combine With Existing HuggingFace Datasets

```python
"""
combine_datasets.py
Merges scraped corpus with existing HuggingFace datasets.
Outputs a single unified JSONL with source metadata preserved.
Downloads HuggingFace datasets and applies same cleaning pipeline.
"""

from datasets import load_dataset
import json, unicodedata
from pathlib import Path
from bnunicodenormalizer import Normalizer

bnorm = Normalizer()

def normalize(text: str) -> str:
    text = bnorm.normalize(text)['normalized']
    return unicodedata.normalize('NFC', text).strip()


OUTPUT = Path("data/final/banglaFM_corpus.jsonl")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

HF_SOURCES = [
    {
        'hf_path': 'uonlp/CulturaX',
        'hf_name': 'bn',
        'text_col': 'text',
        'max_samples': 2_000_000,
        'source_type': 'web_mixed',
        'language_region': 'BD_WB_mix',
    },
    {
        'hf_path': 'oscar-corpus/OSCAR-2301',
        'hf_name': 'bn',
        'text_col': 'text',
        'max_samples': 1_000_000,
        'source_type': 'web_informal',
        'language_region': 'BD_WB_mix',
    },
    {
        'hf_path': 'wikimedia/wikipedia',
        'hf_name': '20231101.bn',
        'text_col': 'text',
        'max_samples': None,   # use all
        'source_type': 'encyclopedic',
        'language_region': 'BD_WB_mix',
    },
    {
        'hf_path': 'HuggingFaceFW/fineweb-edu',
        'hf_name': 'sample-10BT',
        'text_col': 'text',
        'max_samples': 2_000_000,
        'source_type': 'formal_education',
        'language_region': 'EN',
    },
    {
        'hf_path': 'BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset',
        'hf_name': None,
        'text_col': 'review',
        'max_samples': None,
        'source_type': 'code_mixed_commerce',
        'language_region': 'BD_banglish',
    },
]

with open(OUTPUT, 'a', encoding='utf-8') as out_f:
    # First: scraped corpus
    scraped_path = Path("data/cleaned/corpus_decontaminated.jsonl")
    if scraped_path.exists():
        print("Adding scraped corpus...")
        n = 0
        with open(scraped_path) as f:
            for line in f:
                out_f.write(line)
                n += 1
        print(f"  Added {n} scraped documents")
    
    # Then: HuggingFace datasets
    for src in HF_SOURCES:
        print(f"\nLoading {src['hf_path']}...")
        ds = load_dataset(
            src['hf_path'],
            src['hf_name'],
            split='train',
            streaming=True,
            trust_remote_code=True,
        )
        n = 0
        for item in ds:
            text = item.get(src['text_col'], '')
            if not text or len(text.split()) < 20:
                continue
            text = normalize(text)
            record = {
                'source': src['hf_path'],
                'source_type': src['source_type'],
                'language_region': src['language_region'],
                'text': text,
                'word_count': len(text.split()),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
            n += 1
            if src['max_samples'] and n >= src['max_samples']:
                break
        print(f"  Added {n} documents from {src['hf_path']}")

print(f"\nFinal corpus written to {OUTPUT}")
```

---

## PART 3: SYNTHETIC BANGLISH AUGMENTATION

This is your Option 3 novelty claim: first foundation model with explicit, large-scale Banglish pretraining. No prior model has done this intentionally at scale.

### 3.1 Strategy

Three sources of Banglish training data, in order of quality:

1. **Authentic:** BanglishRev reviews (~50M tokens) — real user-written romanized Bangla
2. **Transliterated synthetic:** Take formal Bangla text → transliterate to Latin script → ~0.8B tokens of parallel synthetic Banglish
3. **Transliterated informal:** Same pipeline on informal blog/review text → captures colloquial romanization patterns

### 3.2 Transliteration Pipeline

```python
"""
banglish_augmentation.py
Generates synthetic Banglish from native Bangla text via transliteration.
Uses aksharamukha for Bangla → Latin ISO transliteration.
Outputs parallel corpus: native Bangla + romanized Banglish versions.

Citation in paper: "We augment authentic Banglish data with synthetic
transliteration following TituLLM (Nahin et al., 2025) who generated
3.87B romanized tokens using the same approach."
"""

import json, re
from pathlib import Path
from aksharamukha import transliterate   # pip install aksharamukha

# Alternative: use indic-transliteration
# from indic_transliteration import sanscript
# from indic_transliteration.sanscript import transliterate as itrans

INPUT_CORPUS = Path("data/cleaned/corpus_decontaminated.jsonl")
OUTPUT_BANGLISH = Path("data/banglish/synthetic_banglish.jsonl")
OUTPUT_BANGLISH.parent.mkdir(parents=True, exist_ok=True)

# Only transliterate Bangla-region documents, not English
ELIGIBLE_REGIONS = {'BD', 'BD_WB_mix'}
MAX_DOCS = 1_000_000   # cap at 1M docs → roughly 0.5–1B tokens


def bangla_to_latin(text: str) -> str:
    """
    Transliterate Bangla Unicode text to Latin-script romanization.
    aksharamukha maps Bengali → ISO 15919 (standard scholarly romanization)
    which is close to natural Banglish romanization conventions.
    """
    try:
        return transliterate.process('Bengali', 'ISO', text)
    except Exception:
        return ''


def clean_romanized(text: str) -> str:
    """Post-processing for more natural Banglish appearance."""
    # ISO 15919 uses diacritics (ā, ī, ū, ṭ, ḍ, ṇ, ś, ṣ, ṃ, ḥ)
    # Map common diacritics to ASCII approximations for natural Banglish
    replacements = {
        'ā': 'a', 'ī': 'i', 'ū': 'u',
        'ṭ': 't', 'ḍ': 'd', 'ṇ': 'n',
        'ś': 'sh', 'ṣ': 'sh',
        'ṃ': 'm', 'ḥ': 'h',
        'ṛ': 'ri',
    }
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)
    return text


def main():
    n_processed = 0
    n_written = 0
    
    with open(INPUT_CORPUS) as in_f, open(OUTPUT_BANGLISH, 'w', encoding='utf-8') as out_f:
        for line in in_f:
            if n_processed >= MAX_DOCS:
                break
            try:
                record = json.loads(line)
            except:
                continue
            
            # Only transliterate Bangla-region documents
            if record.get('language_region', '') not in ELIGIBLE_REGIONS:
                continue
            
            text = record.get('text', '')
            if not text or len(text.split()) < 20:
                continue
            
            roman = bangla_to_latin(text)
            roman = clean_romanized(roman)
            if len(roman.split()) < 15:
                continue
            
            banglish_record = {
                'source': record.get('source', record.get('domain', 'scraped')),
                'source_type': 'synthetic_banglish',
                'language_region': 'BD_banglish_synthetic',
                'original_text': text,      # native Bangla (optional, for parallel corpus analysis)
                'text': roman,              # romanized version — this is what the model trains on
                'word_count': len(roman.split()),
                'original_doc_id': record.get('doc_id', ''),
            }
            out_f.write(json.dumps(banglish_record, ensure_ascii=False) + '\n')
            n_written += 1
            n_processed += 1
            
            if n_processed % 10000 == 0:
                print(f"Processed: {n_processed} | Written: {n_written}")
    
    print(f"Done. Generated {n_written} synthetic Banglish documents.")


if __name__ == '__main__':
    main()
```

Install: `pip install aksharamukha`

### 3.3 Banglish Quality Validation

After transliteration, validate that the output looks like realistic Banglish (not garbage):

```python
"""
validate_banglish.py
Spot-checks synthetic Banglish quality.
Prints 50 random samples with their source Bangla for manual inspection.
"""

import json, random
from pathlib import Path

BANGLISH_FILE = Path("data/banglish/synthetic_banglish.jsonl")
N_SAMPLES = 50

records = []
with open(BANGLISH_FILE) as f:
    for line in f:
        try: records.append(json.loads(line))
        except: pass

samples = random.sample(records, min(N_SAMPLES, len(records)))
for i, s in enumerate(samples):
    print(f"\n--- Sample {i+1} ---")
    print(f"BANGLA:   {s.get('original_text', '')[:200]}")
    print(f"BANGLISH: {s['text'][:200]}")
    print(f"Words: {s['word_count']}")
```

Inspect the output manually. If romanization looks unnatural, adjust the `clean_romanized()` diacritic mapping — different Banglish communities (Dhaka vs diaspora) use slightly different conventions.

---

## PART 4: TOKENIZER CORPUS PREPARATION

```python
"""
prepare_tokenizer_corpus.py
Samples from the full corpus proportionally for tokenizer training.
Target: ~10 GB of text, proportional to training mix.
"""

import json, random
from pathlib import Path

CORPUS = Path("data/final/banglaFM_corpus.jsonl")
OUTPUT = Path("data/tokenizer/tokenizer_training_corpus.txt")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# Target sample sizes per source type (in approximate MB of text)
SOURCE_TARGETS_MB = {
    'formal_news': 2000,           # Prothom Alo, BDNews24, etc.
    'web_mixed': 2000,             # CulturaX
    'web_informal': 1000,          # OSCAR
    'informal_blog': 500,          # Somewhereinblog, Sachalayatan
    'informal_commerce_review': 300,
    'formal_official': 200,        # Parliament
    'encyclopedic': 400,           # Wikipedia bn
    'formal_education': 2500,      # FineWeb-Edu (English)
    'code_mixed_commerce': 300,    # BanglishRev
    'synthetic_banglish': 500,     # Synthetic transliterated
    'formal_official': 100,        # Parliament
}

buckets = {k: [] for k in SOURCE_TARGETS_MB}
bucket_sizes = {k: 0 for k in SOURCE_TARGETS_MB}
TARGET_BYTES = {k: v * 1024 * 1024 for k, v in SOURCE_TARGETS_MB.items()}

with open(CORPUS) as f:
    for line in f:
        try:
            record = json.loads(line)
            st = record.get('source_type', 'unknown')
            if st not in buckets: continue
            if bucket_sizes[st] >= TARGET_BYTES[st]: continue
            text = record.get('text', '')
            buckets[st].append(text)
            bucket_sizes[st] += len(text.encode('utf-8'))
        except: continue

print("Bucket sizes:")
for k, v in bucket_sizes.items():
    print(f"  {k}: {v/1024/1024:.1f} MB")

# Write all sampled text to single file, shuffled
all_texts = []
for texts in buckets.values():
    all_texts.extend(texts)
random.shuffle(all_texts)

with open(OUTPUT, 'w', encoding='utf-8') as f:
    for text in all_texts:
        f.write(text.replace('\n', ' ') + '\n')   # one doc per line for spm_train

print(f"\nTokenizer corpus: {OUTPUT}")
print(f"Total lines: {len(all_texts)}")
print(f"Total size: {sum(len(t.encode()) for t in all_texts)/1024/1024/1024:.2f} GB")
```

---

## PART 5: PRETOKENIZATION AND SEQUENCE PACKING

```python
"""
pretokenize_and_pack.py
Converts the full corpus to pretokenized uint16 binary files.
Sequence-packs documents into 2048-token sequences.
This is the final step before training.

Output format: flat binary files, dtype=uint16, shape=(N,)
Each 2048-token chunk is one training sequence.
Documents separated by <eos> token within sequences.
"""

import json, numpy as np
from pathlib import Path
from transformers import PreTrainedTokenizerFast
import sentencepiece as spm

TOKENIZER_PATH = "banglaFM_tokenizer/"
CORPUS = Path("data/final/banglaFM_corpus.jsonl")
OUTPUT_DIR = Path("data/tokenized/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 2048
CHUNK_SIZE = 100_000   # sequences per output shard

tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
EOS_ID = tokenizer.eos_token_id
PAD_ID = tokenizer.pad_token_id


def pack_sequences(token_stream: list[int], seq_len: int) -> np.ndarray:
    """Pack a flat token stream into (N, seq_len) array."""
    # Truncate to multiple of seq_len
    n = (len(token_stream) // seq_len) * seq_len
    return np.array(token_stream[:n], dtype=np.uint16).reshape(-1, seq_len)


def main():
    shard_idx = 0
    token_buffer = []
    total_tokens = 0
    
    with open(CORPUS, encoding='utf-8') as f:
        for doc_idx, line in enumerate(f):
            try:
                record = json.loads(line)
                text = record.get('text', '')
                if not text: continue
                
                # Prepend language token
                lang_region = record.get('language_region', '')
                if 'EN' in lang_region:
                    lang_prefix = '<|lang_en|>'
                elif 'banglish' in lang_region.lower():
                    lang_prefix = '<|lang_bnls|>'
                elif 'mix' in lang_region.lower():
                    lang_prefix = '<|lang_mix|>'
                else:
                    lang_prefix = '<|lang_bn|>'
                
                full_text = f"{lang_prefix} {text}"
                tokens = tokenizer.encode(full_text, add_special_tokens=False)
                tokens.append(EOS_ID)   # document separator
                
                token_buffer.extend(tokens)
                total_tokens += len(tokens)
                
                # When buffer fills CHUNK_SIZE sequences, write a shard
                if len(token_buffer) >= CHUNK_SIZE * SEQ_LEN:
                    packed = pack_sequences(token_buffer, SEQ_LEN)
                    shard_path = OUTPUT_DIR / f"shard_{shard_idx:05d}.npy"
                    np.save(shard_path, packed)
                    print(f"Shard {shard_idx}: {packed.shape[0]} sequences → {shard_path}")
                    token_buffer = token_buffer[CHUNK_SIZE * SEQ_LEN:]
                    shard_idx += 1
                
                if doc_idx % 100_000 == 0:
                    print(f"Docs: {doc_idx} | Total tokens: {total_tokens:,}")
            
            except Exception as e:
                continue
    
    # Write final partial shard
    if token_buffer:
        packed = pack_sequences(token_buffer, SEQ_LEN)
        if packed.shape[0] > 0:
            shard_path = OUTPUT_DIR / f"shard_{shard_idx:05d}.npy"
            np.save(shard_path, packed)
            print(f"Final shard {shard_idx}: {packed.shape[0]} sequences")
    
    print(f"\nPretokenization complete.")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total shards: {shard_idx + 1}")
    print(f"Disk size: ~{total_tokens * 2 / 1024**3:.1f} GB")


if __name__ == '__main__':
    main()
```

---

## PART 6: CORPUS STATISTICS FOR PAPER

```python
"""
corpus_stats.py
Computes and prints statistics for the paper's data section.
Run after cleaning, before training.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

CORPUS = Path("data/final/banglaFM_corpus.jsonl")

stats = defaultdict(lambda: {'docs': 0, 'words': 0, 'chars': 0})
domain_counter = Counter()

with open(CORPUS) as f:
    for line in f:
        try:
            r = json.loads(line)
            st = r.get('source_type', 'unknown')
            text = r.get('text', '')
            stats[st]['docs'] += 1
            stats[st]['words'] += len(text.split())
            stats[st]['chars'] += len(text)
            domain_counter[r.get('domain', r.get('source', 'unknown'))] += 1
        except:
            continue

print("\n=== CORPUS STATISTICS ===\n")
print(f"{'Source Type':<35} {'Docs':>10} {'Words (M)':>12} {'Est. Tokens (M)':>18}")
print("-" * 80)

total_docs, total_words = 0, 0
for st, s in sorted(stats.items(), key=lambda x: -x[1]['words']):
    est_tokens = s['words'] * 1.3   # rough tokens ≈ 1.3× words for Bangla
    print(f"{st:<35} {s['docs']:>10,} {s['words']/1e6:>12.1f} {est_tokens/1e6:>18.1f}")
    total_docs += s['docs']
    total_words += s['words']

print("-" * 80)
print(f"{'TOTAL':<35} {total_docs:>10,} {total_words/1e6:>12.1f} {total_words*1.3/1e6:>18.1f}")
print(f"\nUnique domains/sources: {len(domain_counter)}")
print(f"\nTop 20 domains:")
for domain, count in domain_counter.most_common(20):
    print(f"  {domain}: {count:,} docs")
```

---

## PART 7: EXECUTION ORDER

Run scripts in this exact order:

```bash
# Day 1 — Scraping (runs overnight/multiple days in tmux)
tmux new -s scraper
pip install trafilatura requests beautifulsoup4 lxml pdfplumber fasttext

# Start all scrapers in parallel (separate tmux windows)
tmux new-window -n prothomalo
python prothomalo_scraper.py

tmux new-window -n news
python generic_news_scraper.py

tmux new-window -n blogs
python blog_scraper.py

tmux new-window -n reviews
python rokomari_scraper.py

python parliament_scraper.py   # fast, run inline

# Day 2 — Cleaning (once scraping has substantial data)
pip install bnunicodenormalizer datasketch datasets

# Download fasttext langid model
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

python clean_pipeline.py
python contamination_check.py
python combine_datasets.py       # pulls HuggingFace datasets too
python banglish_augmentation.py  # generate synthetic Banglish
python validate_banglish.py      # MANUALLY INSPECT OUTPUT before proceeding

# Day 2–3 — Tokenizer
python prepare_tokenizer_corpus.py   # sample 10 GB for tokenizer training

spm_train \
  --input=data/tokenizer/tokenizer_training_corpus.txt \
  --model_prefix=banglaFM_tokenizer \
  --vocab_size=32768 \
  [... full command from main guide ...]

# Validate tokenizer fertility
python - << 'EOF'
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("banglaFM_tokenizer.model")
test = "আমি বাংলাদেশের মানুষ। আমি বাংলায় কথা বলি।"
tokens = sp.EncodeAsPieces(test)
print(f"Text: {test}")
print(f"Tokens ({len(tokens)}): {tokens}")
print(f"Fertility: {len(tokens)/len(test.split()):.2f} tokens/word")
# Target: < 2.0 tokens/word for Bangla (GPT-2 gives ~6–8)
EOF

# Day 3 — Pretokenize
python pretokenize_and_pack.py

# Run corpus stats for paper
python corpus_stats.py > corpus_statistics.txt

# Day 3 onward — Training (see main guide)
```

---

## PART 8: DATASHEET FOR PAPER

Include this as an appendix or supplementary material. Required for dataset contribution papers.

```
DATASET NAME: BDCorpus-1B (working name)

MOTIVATION
- Purpose: Pretraining corpus for Bangla foundational language model
- Created by: [Your name], [Your institution]
- Funded by: [Thesis/grant info]

COMPOSITION
- Total documents: [fill after collection]
- Total tokens: [fill after collection]
- Languages: Bangla (Bangladeshi majority ~70%, West Bengali ~30%),
  English (~20% of training mix), Banglish synthetic (~8%)
- Bangladeshi sources: Prothom Alo, BDNews24, Kaler Kantho, Ittefaq,
  Samakal, Somewhereinblog, Sachalayatan, Rokomari, Bangladesh Parliament
- External sources: CulturaX (bn), OSCAR 23.01, Wikipedia (bn), FineWeb-Edu

COLLECTION PROCESS
- Web scraping via trafilatura with robots.txt compliance
- Crawl delay: 1.0–2.0 seconds per request
- Collection period: [dates]
- Preprocessing: bnunicodenormalizer NFC, fasttext langid (>0.80 confidence),
  MinHash LSH deduplication (Jaccard 0.80 threshold, 5-gram shingles)
- Contamination check: 13-gram overlap removal against SentNoB and BLUB

USES
- Intended: NLP research, language model pretraining
- License: CC BY-NC 4.0 (non-commercial research use)
- Not intended for: Commercial deployment, surveillance, misinformation

DISTRIBUTION
- HuggingFace: [your username]/BDCorpus-1B
- Version: 1.0
- DOI: [if available]
```

---

*This plan is designed for Opus to implement from scratch. All scripts are self-contained, resume-safe, and produce documented intermediate artifacts. Run in order. Inspect outputs at each stage before proceeding.*
