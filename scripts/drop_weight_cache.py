#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Drop Linux page cache on model shards before starting Hibana on unified-memory hosts (GB10)."""

import argparse
import os


def drop_path(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        files = [path]
    else:
        files = [
            os.path.join(dirpath, name)
            for dirpath, _dirnames, filenames in os.walk(path)
            for name in filenames
        ]
    count = 0
    for fp in files:
        try:
            fd = os.open(fp, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            count += 1
        except OSError:
            pass
        finally:
            os.close(fd)
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="Extra files or directories to fadvise DONTNEED")
    args = ap.parse_args()

    targets = list(args.paths)
    for env_key in ("HIBANA_FLASHNEXT_CACHE", "HIBANA_W3_CHECKPOINT", "HIBANA_MTP_WEIGHTS"):
        val = os.environ.get(env_key)
        if val:
            targets.append(val)
    results_dir = os.environ.get("HIBANA_RESULTS")
    if results_dir:
        targets.append(os.path.join(results_dir, "w3-checkpoints"))

    total = sum(drop_path(p) for p in dict.fromkeys(targets))
    print(f"fadvise DONTNEED on {total} files")


if __name__ == "__main__":
    main()
