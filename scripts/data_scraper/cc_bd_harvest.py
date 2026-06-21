"""
Common Crawl harvester for Bangladeshi-origin domains -- Bangla script only.

Strategy: Common Crawl already crawled the open web (respecting robots.txt
at crawl time) and explicitly licenses bulk reuse. Instead of scraping
Bangladeshi news sites yourself (ToS risk, server load, legal exposure),
this pulls only the WARC records CC already collected for your domain
allowlist, via HTTP range requests -- no full crawl download needed.

Filters out romanized Bangla (Banglish) -- this pass is Bangla-script only.
Fetch is threaded (I/O-bound: network range requests), extraction runs
across those same threads since trafilatura is the per-doc CPU cost.

pip install requests warcio trafilatura --break-system-packages

Usage:
    python cc_bd_harvest.py --crawl CC-MAIN-2026-22 --out raw/ --workers 24
"""

import argparse
import gzip
import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import trafilatura
from warcio.recordloader import ArcWarcRecordLoader

CC_INDEX_BASE = "https://index.commoncrawl.org"
CC_DATA_BASE = "https://data.commoncrawl.org"

# Unicode Bengali block: U+0980-U+09FF
BENGALI_RE = re.compile(r"[\u0980-\u09FF]")
LATIN_RE = re.compile(r"[a-zA-Z]")


def is_bangla_script(text: str, min_bengali_ratio: float = 0.6) -> bool:
    """
    Reject romanized Bangla / mostly-English text. Counts Bengali-block
    chars vs Latin-block chars among letter characters only (ignores
    digits, punctuation, whitespace so article boilerplate doesn't skew it).
    """
    bengali_chars = len(BENGALI_RE.findall(text))
    latin_chars = len(LATIN_RE.findall(text))
    total_letters = bengali_chars + latin_chars
    if total_letters < 50:
        return False  # too short to judge reliably, skip
    return (bengali_chars / total_letters) >= min_bengali_ratio

# Expand this list as you go. Genre-tagged so you can balance your 6B budget
# across registers (news / govt / informal) rather than just news dominating.
DOMAIN_ALLOWLIST = {
    "news": [
        "prothomalo.com", "bdnews24.com", "kalerkantho.com", "jugantor.com",
        "samakal.com", "ittefaq.com.bd", "banglanews24.com", "jagonews24.com",
        "risingbd.com", "dhakapost.com", "somoynews.tv", "channelionline.com",
    ],
    "government": [
        "bangladesh.gov.bd", "bb.org.bd", "bbs.gov.bd", "nctb.gov.bd",
    ],
    "reference": [
        "bn.banglapedia.org",
    ],
    "informal": [
        "somewhereinblog.net",
    ],
}


def get_latest_crawl_id() -> str:
    """Fetch the most recent crawl identifier from collinfo.json."""
    resp = requests.get(f"{CC_INDEX_BASE}/collinfo.json", timeout=30)
    resp.raise_for_status()
    collections = resp.json()
    return collections[0]["id"]  # newest first


def query_domain(domain: str, crawl_id: str, limit: int = 5000) -> list[dict]:
    """Query the CDX index for all records under a domain in one crawl."""
    url = f"{CC_INDEX_BASE}/{crawl_id}-index"
    params = {
        "url": f"{domain}/*",
        "output": "json",
        "limit": str(limit),
        "filter": "status:200",
    }
    resp = requests.get(url, params=params, timeout=60)
    if resp.status_code == 404:
        return []  # domain not in this crawl
    resp.raise_for_status()
    records = []
    for line in resp.text.strip().split("\n"):
        if line:
            records.append(json.loads(line))
    return records


def fetch_warc_record(record: dict) -> str | None:
    """Range-fetch a single WARC record and extract clean text from it."""
    offset = int(record["offset"])
    length = int(record["length"])
    warc_url = f"{CC_DATA_BASE}/{record['filename']}"
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}

    resp = requests.get(warc_url, headers=headers, timeout=30)
    if resp.status_code not in (200, 206):
        return None

    stream = io.BytesIO(gzip.decompress(resp.content))
    loader = ArcWarcRecordLoader()
    warc_record = loader.parse_record_stream(stream)
    html = warc_record.content_stream().read()

    text = trafilatura.extract(html, include_comments=False, include_tables=False)
    if not text or len(text) < 200:
        return None
    if not is_bangla_script(text):
        return None  # romanized / mostly-English -- not this pass
    return text


def _process_record(rec: dict, genre: str, domain: str) -> dict | None:
    """Worker function: fetch + extract + filter one record."""
    text = fetch_warc_record(rec)
    if not text:
        return None
    return {
        "domain": domain,
        "genre": genre,
        "url": rec.get("url"),
        "timestamp": rec.get("timestamp"),
        "text": text,
    }


def harvest(crawl_id: str, out_dir: Path, per_domain_limit: int = 2000,
            workers: int = 16) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()

    for genre, domains in DOMAIN_ALLOWLIST.items():
        out_path = out_dir / f"{genre}.jsonl"
        kept_total = 0

        for domain in domains:
            print(f"[{genre}] querying {domain} in {crawl_id} ...")
            records = query_domain(domain, crawl_id, limit=per_domain_limit)
            print(f"  -> {len(records)} candidate records, fetching with {workers} workers")

            kept = 0
            with ThreadPoolExecutor(max_workers=workers) as pool, \
                 out_path.open("a", encoding="utf-8") as f:
                futures = {
                    pool.submit(_process_record, rec, genre, domain): rec
                    for rec in records
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result is None:
                        continue
                    with write_lock:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    kept += 1

            print(f"  -> kept {kept}/{len(records)} (Bangla-script, length-filtered)")
            kept_total += kept

        print(f"[{genre}] total kept: {kept_total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crawl", default=None,
                         help="Crawl ID e.g. CC-MAIN-2026-22; defaults to latest")
    parser.add_argument("--out", default="raw_bd_corpus")
    parser.add_argument("--per-domain-limit", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=24,
                         help="Thread pool size. I/O-bound (network range "
                              "requests), so this can exceed core count -- "
                              "20-32 is reasonable on an 8-core 7700; trafilatura "
                              "parsing is the CPU cost per doc, not the fetch.")
    args = parser.parse_args()

    crawl_id = args.crawl or get_latest_crawl_id()
    print(f"Using crawl: {crawl_id}")
    harvest(crawl_id, Path(args.out), per_domain_limit=args.per_domain_limit,
            workers=args.workers)
