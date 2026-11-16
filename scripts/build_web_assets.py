"""Assemble the files the Korean browser demo serves, into web/public/assets.

    python scripts/build_web_assets.py [--export]

The page runs the phonemizer in Pyodide and the model in ONNX Runtime Web, so
everything has to be a plain file the browser can fetch:

| asset | from |
| --- | --- |
| ``model.onnx``, ``config.json`` | ``runs/full/inference.pt`` through ``tiny_voca.export`` |
| ``phonemizer/*`` | ``weights/phonemizer`` (dizione's built dictionaries) |
| ``phonemizer/cmudict.zip`` | the NLTK corpus g2pK reads for English words |
| ``mecab-dict.json`` | ``scripts/build_mecab_dict.py`` (MeCab analyses of the corpus) |
| ``dizione.zip`` | ``dizione/src/dizione`` — the English side, unzipped into Pyodide |
| ``pyfront.zip`` | g2pK + jamo + the ``kofront`` package (vocab/text/korean) + the MeCab shim |

``--export`` re-runs the ONNX export; otherwise an existing ``model.onnx`` is kept.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "tiny-voca" / "web" / "public" / "assets"
DIZIONE_DATA = ROOT / "weights" / "phonemizer"
SITE_PACKAGES = ROOT / ".venv" / "Lib" / "site-packages"
FRONTEND_FILES = ("vocab.py", "text.py", "korean.py")
DIZIONE_BUILT = ("lexicon.tsv", "align.json", "g2p.json", "g2p.npz", "homographs.json")


def zip_tree(target: Path, base: Path, prefix: Path = Path(".")) -> int:
    """Zip ``base`` into ``target`` under ``prefix``, without __pycache__."""
    target.parent.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(base.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, (prefix / path.relative_to(base)).as_posix())
    return len(files)


def zip_kofront(target: Path) -> int:
    """g2pK + jamo + the front-end package + the MeCab shim, one zip for Pyodide."""
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for package in ("g2pk", "jamo"):
            for path in sorted((SITE_PACKAGES / package).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, f"{package}/{path.relative_to(SITE_PACKAGES / package)}")
                    count += 1
        source = ROOT / "tiny-voca" / "src" / "tiny_voca"
        archive.writestr("kofront/__init__.py",
                         "from .korean import phonemize_mixed, expand_numbers  # noqa: F401\n")
        count += 1
        for name in FRONTEND_FILES:
            archive.write(source / name, f"kofront/{name}")
            count += 1
        archive.write(ROOT / "scripts" / "mecab_shim.py", "mecab.py")     # python-mecab-ko stand-in
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "full" / "inference.pt")
    parser.add_argument("--export", action="store_true", help="re-run tiny_voca.export")
    args = parser.parse_args()

    ASSETS.mkdir(parents=True, exist_ok=True)
    onnx = ASSETS / "model.onnx"
    if args.export or not onnx.is_file():
        subprocess.run([sys.executable, "-m", "tiny_voca.export",
                        "--checkpoint", str(args.checkpoint), "--output", str(onnx)],
                       check=True, cwd=ROOT)
    metadata = onnx.with_suffix(".json")
    if not metadata.is_file():
        raise SystemExit(f"run the export first: {metadata} is missing")
    (ASSETS / "config.json").write_text(metadata.read_text(encoding="utf-8"), encoding="utf-8")

    (ASSETS / "phonemizer").mkdir(parents=True, exist_ok=True)
    for name in DIZIONE_BUILT:
        shutil.copy2(DIZIONE_DATA / name, ASSETS / "phonemizer" / name)

    cmudict = next((p for p in [
        Path.home() / "AppData" / "Roaming" / "nltk_data" / "corpora" / "cmudict.zip",
        Path.home() / "nltk_data" / "corpora" / "cmudict.zip",
    ] if p.is_file()), None)
    if cmudict is None:
        raise SystemExit("cmudict.zip not found: python -m nltk.cli? (nltk.download('cmudict'))")
    shutil.copy2(cmudict, ASSETS / "phonemizer" / "cmudict.zip")

    dictionary = ASSETS / "mecab-dict.json"
    if not dictionary.is_file():
        raise SystemExit("mecab-dict.json not found: run scripts/build_mecab_dict.py")

    dizione_files = zip_tree(ASSETS / "dizione.zip", ROOT / "dizione" / "src" / "dizione", Path("dizione"))
    front_files = zip_kofront(ASSETS / "pyfront.zip")

    total = sum(p.stat().st_size for p in ASSETS.rglob("*") if p.is_file())
    print(f"assets -> {ASSETS}")
    for path in sorted(ASSETS.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(ASSETS).as_posix():34s} {path.stat().st_size / 1e6:8.2f} MB")
    print(f"dizione.zip {dizione_files} files, pyfront.zip {front_files} files")
    print(f"total {total / 1e6:.1f} MB to download on the first visit")


if __name__ == "__main__":
    main()
