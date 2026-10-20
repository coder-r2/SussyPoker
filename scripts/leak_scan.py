"""Leak guard (SPEC C5): fail if any tracked file or web export contains a competition-ID-shaped string.

    python scripts/leak_scan.py            # exit 1 on any match; prints file names and counts only, never the match

Scans every file git tracks plus everything under web/public/data (tracked or not), so it also catches a fresh
export before it is committed. Used by CI and by tests/test_leak_guard.py.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from suspoker.export_web import ID_PATTERN  # noqa: E402

SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".pkl", ".npy", ".parquet"}


def files_to_scan() -> list[Path]:
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    paths = {ROOT / p for p in tracked.decode("utf-8").split("\0") if p}
    paths |= set((ROOT / "web" / "public" / "data").rglob("*"))
    return sorted(p for p in paths if p.is_file() and p.suffix.lower() not in SKIP_SUFFIXES)


def scan() -> dict[str, int]:
    hits = {}
    for p in files_to_scan():
        n = len(ID_PATTERN.findall(p.read_text(encoding="utf-8", errors="ignore")))
        if n:
            hits[str(p.relative_to(ROOT))] = n
    return hits


def main() -> int:
    hits = scan()
    for path, n in hits.items():
        print(f"LEAK: {path}: {n} competition-ID-shaped strings")
    print(f"leak scan: {len(files_to_scan())} files, {len(hits)} with matches")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
