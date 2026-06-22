"""
Targeted Common Crawl WET harvester — downloads only segments with BD content.

Step 1: Downloads WET paths list, filters to known BD segments
Step 2: Downloads matching WET files in parallel (streaming, not saved to disk)
Step 3: Parses pre-extracted text, filters by domain allowlist
Step 4: Saves per-domain JSONL files

Much faster than per-record CDX approach. No trafilatura needed.

Usage:
    python scripts/wet_harvest.py --out saved/data/raw/cc_bangla/ --workers 32
"""

import argparse
import gzip
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

CC_BASE = "https://data.commoncrawl.org"
BENGALI_RE = re.compile(r"[\u0980-\u09FF]")
LATIN_RE = re.compile(r"[a-zA-Z]")

# Segment IDs known to contain BD content (from CDX probe of 54 BD domains)
BD_SEGMENTS = {
    "1778213376756.47", "1778213376806.31", "1778213376840.35",
    "1778213376873.15", "1778213376876.49", "1778213376901.20",
    "1778213376939.73", "1778213376959.33", "1778213376970.26",
    "1778213376977.1", "1778213376977.44", "1778213377023.25",
    "1778213377043.87", "1778213377075.59", "1778213377086.95",
    "1778213377098.93", "1778213377106.75", "1778213377112.37",
    "1778213377120.66", "1778213377132.38", "1778213377160.14",
    "1778213377162.4", "1778213377163.77", "1778213377171.10",
    "1778213377174.53", "1778213377191.12", "1778213377198.32",
    "1778213377201.76", "1778213377248.11", "1778213377249.17",
    "1778213377272.68", "1778213377288.18", "1778213377298.13",
    "1778213377303.65", "1778213377304.58", "1778213377321.70",
    "1778213377336.64", "1778213377341.24", "1778213377345.79",
    "1778213377345.9", "1778213377370.50", "1778213377372.85",
    "1778213377391.46", "1778213377403.40", "1778213377432.7",
    "1778213377435.62", "1778213377442.39", "1778213377458.28",
    "1778213377469.30", "1778213377526.23", "1778213377534.41",
    "1778213377557.42", "1778213377561.52", "1778213377585.61",
    "1778213377589.60", "1778213377596.29", "1778213377607.45",
    "1778213377616.21", "1778213377623.34", "1778213377624.36",
    "1778213377631.43", "1778213377646.27", "1778213377649.69",
    "1778213377650.51", "1778213377652.5", "1778213377666.57",
    "1778213377724.56", "1778213377776.6", "1778213377796.94",
    "1778213377820.54", "1778213377991.19", "1778213378037.67",
    "1778213378173.48", "1778213378177.22", "1778213378179.2",
    "1778213378195.55", "1778213378292.3", "1778213378318.0",
    "1778213378330.16", "1778213378386.8", "1778213672391.90",
    "1778213673305.63", "1778213676796.71", "1778213677484.92",
    "1778213677977.72", "1778213678100.81", "1778213678424.99",
    "1778213678516.83", "1778213678841.80", "1778213678843.89",
    "1778213679070.78", "1778213679103.88", "1778213679840.86",
    "1778213680053.74", "1778213680199.98", "1778213680237.91",
    "1778213680315.97", "1778213681280.82", "1778213681280.84",
    "1778213681532.96",
}

DOMAINS = [
    "prothomalo.com", "kalerkantho.com", "jugantor.com", "samakal.com",
    "ittefaq.com.bd", "bdnews24.com", "banglanews24.com", "jagonews24.com",
    "risingbd.com", "dhakapost.com", "somoynews.tv", "channelionline.com",
    "bd-pratidin.com", "amadershomoy.com", "independent24.com", "ntvbd.com",
    "jamunatv.com", "news24bd.tv", "ekushey-tv.com", "rtvbd.com",
    "dailystarbangla.com", "thefinancialexpress.com.bd",
    "deshrupantor.com", "banglatribune.com", "kalbela.com", "manabzamin.com",
    "bhorerkagoj.com", "nayadiganta.com", "protidinersangbad.com",
    "ajkerpordomoy.com", "dailysangbad.com.bd", "dainandin.com",
    "shomoyeralo.com", "bd24live.com", "mzamin.com", "newsbangla24.com",
    "goshinews.com", "thebangladeshtoday.com", "barta24.com", "ekattor.tv",
    "nagoriknews.tv", "alcnews.com",
    "blitzbd.com", "dhakatimes24.com", "khaborer-kagoj.com",
    "lalonmart.com", "banglaonline.com",
    "bangladesh.gov.bd", "bb.org.bd", "bbs.gov.bd", "nctb.gov.bd",
    "moedu.gov.bd", "mopa.gov.bd", "bdgovt.com",
    "bn.banglapedia.org", "bn.wikipedia.org",
    "somewhereinblog.net", "valoidea.com", "banglarsamaj.com",
]

DOMAIN_SET = set(DOMAINS)


def is_bangla(text: str, min_ratio: float = 0.6) -> bool:
    bengali = len(BENGALI_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    total = bengali + latin
    if total < 50:
        return False
    return (bengali / total) >= min_ratio


def match_domain(url: str) -> str | None:
    url_lower = url.lower()
    for d in DOMAIN_SET:
        if d in url_lower:
            return d
    return None


def parse_wet_bytes(raw: bytes) -> list[dict]:
    """Parse gzipped WET content, return matching Bangla docs."""
    try:
        text = gzip.decompress(raw).decode("utf-8", errors="replace")
    except Exception:
        return []

    docs = []
    current_uri = None
    content_lines = []

    for line in text.split("\n"):
        if line.startswith("WARC-Target-URI:"):
            # Save previous record
            if current_uri and content_lines:
                content = "\n".join(content_lines).strip()
                domain = match_domain(current_uri)
                if domain and len(content) >= 200 and is_bangla(content):
                    docs.append({
                        "domain": domain,
                        "url": current_uri,
                        "text": content,
                    })
            current_uri = line.split(":", 1)[1].strip()
            content_lines = []
        elif line.strip() == "" and current_uri is not None:
            # Blank line = separator between header and content
            content_lines = []
        elif not line.startswith("WARC-") and not line.startswith("Content-") and current_uri is not None:
            content_lines.append(line)

    # Last record
    if current_uri and content_lines:
        content = "\n".join(content_lines).strip()
        domain = match_domain(current_uri)
        if domain and len(content) >= 200 and is_bangla(content):
            docs.append({
                "domain": domain,
                "url": current_uri,
                "text": content,
            })

    return docs


def download_wet(path: str, timeout: int = 60) -> list[dict]:
    url = f"{CC_BASE}/{path}"
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return parse_wet_bytes(resp.content)
            elif resp.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            return []
        except Exception:
            time.sleep(5 * (attempt + 1))
    return []


def main():
    parser = argparse.ArgumentParser(description="Targeted BD WET harvester")
    parser.add_argument("--out", default="saved/data/raw/cc_bangla/")
    parser.add_argument("--crawl", default="CC-MAIN-2026-21")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get WET paths and filter to BD segments
    print("Downloading WET paths list...", flush=True)
    paths_url = f"https://data.commoncrawl.org/crawl-data/{args.crawl}/wet.paths.gz"
    resp = requests.get(paths_url, timeout=30)
    all_paths = gzip.decompress(resp.content).decode().strip().split("\n")
    print(f"Total WET files in crawl: {len(all_paths)}", flush=True)

    # Filter to BD segments
    bd_paths = []
    for p in all_paths:
        for seg in BD_SEGMENTS:
            if seg in p:
                bd_paths.append(p)
                break

    print(f"BD segment WET files: {len(bd_paths)}", flush=True)
    print(f"Estimated download: {len(bd_paths) * 61 / 1024:.0f} GB", flush=True)

    if not bd_paths:
        print("No matching WET files found!", flush=True)
        return

    # Step 2: Download, parse, filter
    out_files = {}
    write_lock = threading.Lock()
    total_kept = 0
    start_time = time.time()

    def flush_doc(doc):
        nonlocal total_kept
        domain = doc["domain"]
        with write_lock:
            if domain not in out_files:
                out_files[domain] = open(
                    out_dir / f"{domain.replace('.', '_')}.jsonl",
                    "w", encoding="utf-8",
                )
            out_files[domain].write(json.dumps(doc, ensure_ascii=False) + "\n")
            total_kept += 1

    pbar = tqdm(total=len(bd_paths), desc="WET files", unit="file", ncols=100)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_wet, p, args.timeout): p for p in bd_paths}
        for future in as_completed(futures):
            docs = future.result()
            for doc in docs:
                flush_doc(doc)
            pbar.update(1)
            pbar.set_postfix(kept=total_kept)

    pbar.close()
    elapsed = time.time() - start_time

    for f in out_files.values():
        f.close()

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"Total kept: {total_kept:,}", flush=True)
    print(f"Domains with data: {len(out_files)}", flush=True)
    for domain in sorted(out_files.keys()):
        fpath = out_dir / f"{domain.replace('.', '_')}.jsonl"
        lines = sum(1 for _ in open(fpath))
        print(f"  {domain}: {lines:,} docs", flush=True)


if __name__ == "__main__":
    main()
