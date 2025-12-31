"""Download NINAPRO Preprocessed files for a chosen database (DB1-DB4)
and extract into data/raw/DB<db> by default.

Examples:
    # Download DB1 subjects 1-27 into data/raw/DB1 (default)
    python scripts/download_data.py --db 1 --start 1 --end 27 --workers 4

    # Download DB2 subjects 1-40 into data/raw/DB2
    python scripts/download_data.py --db 2 --start 1 --end 40

    # Override destination and base URL explicitly
    python scripts/download_data.py --db 3 --dest my_data/DB3 --base-url https://ninapro.hevs.ch/files/DB3/Preprocessed/

Features:
- Choose database via --db (1-4); constructs default base URL and destination.
- Skips files already downloaded (compares Content-Length when available).
- Concurrent downloads with ThreadPoolExecutor.
- Extracts zip files into dedicated subject subfolders (sX/).
- Optional --keep-zip to retain the zip files.

Dependencies: requests, tqdm
"""

from __future__ import annotations

import argparse
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

DEST_TEMPLATE = "data/raw/DB{db}"
DEFAULT_DB = "1"
DB_CONFIG = {
    "1": {
        "base_url": "https://ninapro.hevs.ch/files/DB1/Preprocessed/",
        "file_name": "s{index}.zip",
        "start": 1,
        "end": 27,
    },
    "2": {
        "base_url": "https://ninapro.hevs.ch/files/DB2_Preproc/",
        "file_name": "DB2_s{index}.zip",
        "start": 1,
        "end": 40,
    },
    "3": {
        "base_url": "https://ninapro.hevs.ch/files/db3_Preproc/",
        "file_name": "s{index}_0.zip",
        "start": 1,
        "end": 11,
    },
    "4": {
        "base_url": "https://ninapro.hevs.ch/files/DB4_Preproc/",
        "file_name": "s{index}.zip",
        "start": 1,
        "end": 10,
    },
}


def download_file(
    url: str,
    out_path: Path,
    timeout: int = 30,
    retries: int = 3,
    chunk_size: int = 1024 * 32,
) -> bool:
    """Download single file with simple retry logic. Returns True on success."""
    out_path = Path(out_path)
    tmp_path = out_path.with_suffix(".part")

    for attempt in range(1, retries + 1):
        try:
            # HEAD request for Content-Length (if available)
            total = None
            try:
                head = requests.head(url, timeout=timeout)
                if head.status_code == 200 and "Content-Length" in head.headers:
                    total = int(head.headers["Content-Length"])
            except Exception:
                total = None

            # If final file exists and size matches header, skip
            if out_path.exists() and total is not None and out_path.stat().st_size == total:
                print(f"Skipping existing: {out_path.name} (size matches)")
                return True

            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                total_bytes = int(r.headers.get("Content-Length", 0)) or None

                pbar = tqdm(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    desc=out_path.name,
                    leave=False,
                )

                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        pbar.update(len(chunk))
                pbar.close()

            # Move tmp to final
            tmp_path.replace(out_path)

            # Final size check if possible
            if total is not None and out_path.stat().st_size != total:
                raise IOError("Downloaded file size does not match Content-Length")

            return True

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Attempt {attempt} failed for {url}: {e}")
            time.sleep(2**attempt)

    print(f"Failed to download {url} after {retries} attempts")
    return False


def extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest_dir)
        return True
    except zipfile.BadZipFile:
        print(f"Bad zip file: {zip_path}")
        return False


def download_range(
    dest: Path,
    base_url: str,
    file_name: str,
    start: int = 1,
    end: int = 27,
    workers: int = 4,
    keep_zip: bool = False,
) -> None:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    tasks = []
    for i in range(start, end + 1):
        name = file_name.format(index=i)
        url = base_url.rstrip("/") + "/" + name
        out_file = dest / name
        tasks.append((i, url, out_file))

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_file, url, out): (i, url, out) for i, url, out in tasks}
        for fut in as_completed(futures):
            i, url, out = futures[fut]
            try:
                ok = fut.result()
            except Exception as e:
                print(f"Download error for {url}: {e}")
                ok = False
            results.append((i, url, out, ok))

    # Extract into dedicated sX/ subfolders
    for i, url, out, ok in results:
        if not ok:
            print(f"Skipping extraction (download failed): {out.name}")
            continue

        subject_dir = dest / f"s{i}"
        subject_dir.mkdir(exist_ok=True)

        print(f"Extracting {out.name} to {subject_dir}")
        ok2 = extract_zip(out, subject_dir)

        if ok2 and not keep_zip:
            try:
                out.unlink()
            except Exception as e:
                print(f"Could not remove zip {out}: {e}")

    print("Done.")


def normalize_db(db: str) -> str:
    """Normalize db string: accept '1', 'DB1', 'db1' etc., return just the number as string."""
    db = db.strip()
    if db.lower().startswith("db"):
        db = db[2:]
    if db.isdigit():
        db = str(int(db))
    return db


def get_db_config(db_num: str) -> dict:
    if db_num.upper() == "X":
        raise NotImplementedError("DBX is not implemented")
    if db_num not in DB_CONFIG:
        raise NotImplementedError(f"DB{db_num} is not implemented")
    return DB_CONFIG[db_num]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Download NINAPRO Preprocessed files for a chosen database (DB1-DB4) "
            "and extract into data/raw/DB<db> by default."
        )
    )

    p.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB,
        help="NINAPRO database to download (1-4, or e.g. 'DB1'). Default: 1",
    )

    p.add_argument(
        "--dest",
        default=None,
        help=(
            "Destination directory for downloaded zips and extracted files "
            "(default: data/raw/DB<db>)"
        ),
    )
    p.add_argument(
        "--base-url",
        default=None,
        help=(
            "Base URL for files. If not provided, defaults to the DB-specific URL."
        ),
    )
    p.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index (inclusive). Default depends on DB.",
    )
    p.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (inclusive). Default depends on DB.",
    )
    p.add_argument("--workers", type=int, default=4, help="Number of concurrent downloads")
    p.add_argument("--keep-zip", action="store_true", help="Keep downloaded zip files after extraction")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    db_num = normalize_db(args.db)
    cfg = get_db_config(db_num)

    # Determine base URL
    if args.base_url is None:
        base_url = cfg["base_url"]
    else:
        base_url = args.base_url

    # Determine destination directory
    if args.dest is None:
        dest = Path(DEST_TEMPLATE.format(db=db_num))
    else:
        dest = Path(args.dest)

    if args.start is None:
        start = cfg["start"]
    else:
        start = args.start

    if args.end is None:
        end = cfg["end"]
    else:
        end = args.end

    download_range(
        dest,
        base_url,
        cfg["file_name"],
        start=start,
        end=end,
        workers=args.workers,
        keep_zip=args.keep_zip,
    )


if __name__ == "__main__":
    main()
