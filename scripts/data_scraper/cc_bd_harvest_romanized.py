"""
Common Crawl harvester for romanized Bangla (Banglish) text.

Companion to cc_bd_harvest.py, which deliberately excludes this. Same
mechanism (CDX index -> range-fetch WARC -> trafilatura extraction),
different filter and different domain targets.

Romanized Bangla doesn't show up much on formal news/government domains --
those write in Bangla script. It lives in informal/social text: blogs,
forum posts, comment sections. Domain list below reflects that; expand
with more Bangladeshi blog platforms and forums as you find them.

Filtering is harder here than the Bangla-script pass: "mostly Latin
characters" alone just matches English. This uses a small wordlist of
common high-frequency Bangla function/discourse words in their romanized
forms (ami, tumi, kintu, na, hocche, etc.) and requires a minimum hit
rate -- crude but cheap, and good enough as a first-pass filter. Treat
this corpus as noisier than the Bangla-script one and plan a manual
spot-check pass before it goes anywhere near the tokenizer.

pip install requests warcio trafilatura --break-system-packages

Usage:
    python cc_bd_harvest_romanized.py --crawl CC-MAIN-2026-22 --out raw_romanized/ --workers 24
"""

import argparse
import gzip
import io
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import trafilatura
from warcio.recordloader import ArcWarcRecordLoader

CC_INDEX_BASE = "https://index.commoncrawl.org"
CC_DATA_BASE = "https://data.commoncrawl.org"

BENGALI_RE = re.compile(r"[\u0980-\u09FF]")
LATIN_RE = re.compile(r"[a-zA-Z]")

# High-frequency romanized Bangla function/discourse words. Word-boundary
# matched, case-insensitive. This is a starter set -- the more of these
# you add (and the more dialectal/regional spelling variants), the better
# the filter gets. Crude on purpose: a real classifier is a later-stage
# problem, this is just keeping obvious junk out of the raw pull.
ROMANIZED_BANGLA_MARKERS = [
    "ami", "tumi", "apni", "amra", "tara", "amar", "tomar", "tar",
    "kintu", "kemon", "kothay", "kivabe", "keno", "ki", "na", "hae",
    "hocche", "korbo", "korlam", "hoyeche", "thik", "ache", "nei",
    "valo", "bhalo", "kharap", "onek", "ektu", "sudhu", "shudhu",
    "tahole", "ekhon", "ajke", "kalke", "biye", "bondhu", "bhai",
    "apu", "dada", "didi", "shob", "sob", "jonno", "diye", "kore",
]
MARKER_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in ROMANIZED_BANGLA_MARKERS) + r")\b",
    re.IGNORECASE,
)


def is_romanized_bangla(text: str, min_latin_ratio: float = 0.7,
                         min_marker_hits_per_500_chars: float = 1.0) -> bool:
    """
    Reject pure Bangla-script text (belongs in the other pass) and reject
    plain English (no Bangla markers). Keep text that's mostly Latin
    script AND has a reasonable density of romanized Bangla function words.
    """
    bengali_chars = len(BENGALI_RE.findall(text))
    latin_chars = len(LATIN_RE.findall(text))
    total_letters = bengali_chars + latin_chars
    if total_letters < 50:
        return False

    if bengali_chars / total_letters > 0.1:
        return False  # has real Bangla script -- not what this pass wants

    if (latin_chars / total_letters) < min_latin_ratio:
        return False

    marker_hits = len(MARKER_RE.findall(text))
    expected_hits = (len(text) / 500) * min_marker_hits_per_500_chars
    return marker_hits >= max(expected_hits, 2)  # absolute floor of 2 hits


# Romanized Bangla lives in informal/social text, not formal news/gov sites.
# Expand this aggressively -- this list is a starting point, not a finished
# allowlist. Bangladeshi Facebook-adjacent blog platforms, forums, and
# comment-heavy sites are where you'll find real density.
DOMAIN_ALLOWLIST = {
    "blogs_forums": [
        "somewhereinblog.net",
        "forum.projuktiteam.com",
    ],
    "news_comments": [
        # Comment sections on news sites often carry romanized Bangla even
        # though the articles themselves don't -- worth a pass, but expect
        # a much lower hit rate than dedicated blog/forum platforms.
        "prothomalo.com",
        "bdnews24.com",
    ],
}


def get_latest_crawl_id() -> str:
    resp = requests.get(f"{CC_INDEX_BASE}/collinfo.json", timeout=30)
    resp.raise_for_status()
    return resp.json()[0]["id"]


def query_domain(domain: str, crawl_id: str, limit: int = 5000) -> list[dict]:
    url = f"{CC_INDEX_BASE}/{crawl_id}-index"
    params = {
        "url": f"{domain}/*",
        "output": "json",
        "limit": str(limit),
        "filter": "status:200",
    }
    resp = requests.get(url, params=params, timeout=60)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.strip().split("\n") if line]


def fetch_warc_record(record: dict) -> str | None:
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

    text = trafilatura.extract(html, include_comments=True, include_tables=False)
    # include_comments=True here (unlike the Bangla-script script) -- for
    # the news_comments domains the comment text is the actual target,
    # not noise to strip.
    if not text or len(text) < 100:
        return None
    if not is_romanized_bangla(text):
        return None
    return text


def _process_record(rec: dict, genre: str, domain: str) -> dict | None:
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

            print(f"  -> kept {kept}/{len(records)} (romanized-Bangla filtered)")
            kept_total += kept

        print(f"[{genre}] total kept: {kept_total}")
        if genre == "news_comments" and kept_total < 50:
            print("  (low yield expected here -- comment sections are a thin "
                  "source, this is a supplementary pass, not your bulk volume)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crawl", default=None,
                         help="Crawl ID e.g. CC-MAIN-2026-22; defaults to latest")
    parser.add_argument("--out", default="raw_romanized_corpus")
    parser.add_argument("--per-domain-limit", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    crawl_id = args.crawl or get_latest_crawl_id()
    print(f"Using crawl: {crawl_id}")
    harvest(crawl_id, Path(args.out), per_domain_limit=args.per_domain_limit,
            workers=args.workers)
