"""Stage prepared OmniLearned shards or bounded row samples without ML installs.

Run on a Slurm compute node in the unchanged CMS Pixi environment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone
from urllib.parse import urljoin

import h5py
import requests

from inspect_omnilearned import HTTPRange

BASE = "https://portal.nersc.gov/cfs/dasrepo/omnilearned/"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=["aspen", "jetclass", "jetclass2"], default=["aspen", "jetclass", "jetclass2"])
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    parser.add_argument("--max-files", type=int, default=1, help="Per split; 0 means all")
    parser.add_argument("--rows", type=int, default=1024, help="First rows per shard; 0 downloads full shards")
    args = parser.parse_args()
    if args.rows < 0 or args.max_files < 0:
        parser.error("limits must be nonnegative")
    args.root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    manifest = {"run_id": run_id, "status": "running", "question": "Can prepared public shards load into heptokens preserving splits and labels?",
                "success_criterion": "Finite features, upstream-compatible mask, fixed labels, and a batch from each split",
                "args": vars(args) | {"root": str(args.root)}, "seed": None, "checkpoint": None,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "files": [],
                "revision": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                "lock_sha256": sha256(repo / "environments/cms-gpu/pixi.lock")}
    record = args.root / f"stage-{run_id}.json"
    record.write_text(json.dumps(manifest, indent=2))
    (args.root / f"changes-{run_id}.diff").write_text(subprocess.check_output(["git", "-C", str(repo), "diff", "--", "src", "tools", "tests"], text=True))
    try:
        for dataset in args.datasets:
            for split in args.splits:
                directory = urljoin(BASE, f"{dataset}/{split}/")
                listing = requests.get(directory, timeout=60)
                listing.raise_for_status()
                names = sorted(set(re.findall(r'href="([^"/]+\.h5)"', listing.text)))
                if not names:
                    raise ValueError(f"No files at {directory}")
                if args.max_files:
                    names = names[:args.max_files]
                destination = args.root / dataset / split
                destination.mkdir(parents=True, exist_ok=True)
                for name in names:
                    url = urljoin(directory, name)
                    output = destination / name
                    if output.exists():
                        raise FileExistsError(f"Use a fresh root to avoid stale/overwritten inputs: {output}")
                    temporary = output.with_suffix(".partial")
                    entry = {"url": url, "output": str(output), "row_selection": [0, args.rows] if args.rows else "all"}
                    print(f"Staging {url}", flush=True)
                    if args.rows:
                        with HTTPRange(url) as stream, h5py.File(stream, "r") as source, h5py.File(temporary, "w") as target:
                            entry.update(source_bytes=stream.size, source_etag=stream.etag,
                                         source_last_modified=stream.last_modified,
                                         source_rows=len(source["data"]))
                            rows = min(args.rows, len(source["data"]))
                            for key in ("data", "global", "pid"):
                                target.create_dataset(key, data=source[key][:rows], compression="gzip")
                            target.attrs["source_url"] = url
                            target.attrs["row_start"] = 0
                            target.attrs["row_stop"] = rows
                            target.attrs["sample_only"] = True
                            entry["rows"] = rows
                    else:
                        with requests.get(url, stream=True, timeout=120) as response:
                            response.raise_for_status()
                            entry["source_etag"] = response.headers.get("ETag")
                            with open(temporary, "wb") as stream:
                                for chunk in response.iter_content(1024*1024):
                                    stream.write(chunk)
                        with h5py.File(temporary, "r") as source:
                            entry["rows"] = len(source["data"])
                            for key in ("global", "pid"):
                                if len(source[key]) != entry["rows"]:
                                    raise ValueError("Mismatched rows")
                    temporary.rename(output)
                    entry.update(sha256=sha256(output), bytes=output.stat().st_size)
                    manifest["files"].append(entry)
                    record.write_text(json.dumps(manifest, indent=2))
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        record.write_text(json.dumps(manifest, indent=2))
    print(record, flush=True)


if __name__ == "__main__":
    main()
