import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _leak_scan():
    spec = importlib.util.spec_from_file_location("leak_scan", ROOT / "scripts" / "leak_scan.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_repository_and_web_export_contain_no_competition_ids():
    hits = _leak_scan().scan()
    assert hits == {}, f"competition-ID-shaped strings found in: {sorted(hits)}"


def test_leak_scan_covers_the_web_export():
    files = _leak_scan().files_to_scan()
    web = ROOT / "web" / "public" / "data"
    if (web / "summary.json").exists():
        assert web / "summary.json" in files
