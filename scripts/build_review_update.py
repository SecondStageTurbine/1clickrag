"""Build a source-only overlay and verify its extracted integration tests.

Explicit file allowlists keep corpora, local configuration, indexes and models
out. Existing archives are never overwritten. Run using the installed venv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


ROOT_FILES = (
    "README.md", "REVIEW-MODES-UPDATE.md", "LICENSE", ".env.example",
    "requirements.txt", "requirements-native.txt", "requirements-gpu.txt",
    "Rag.bat", "rag-up.ps1", "rag-up.sh", "rag-client.psm1",
    "Dockerfile", "docker-compose.yml", "selftest.py",
    "rag-eval.example.json", "rag-projects.example.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="rag-review-modes-update-v12.zip")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    files = [root / name for name in ROOT_FILES]
    files += sorted(p for p in (root / "app").rglob("*")
                    if p.is_file() and p.suffix in {".py", ".txt", ".html", ".js", ".css"}
                    and "__pycache__" not in p.parts)
    files += sorted(p for p in (root / "tests").iterdir()
                    if p.is_file() and p.suffix in {".py", ".cjs"})
    files.append(Path(__file__).resolve())
    contents = {p.relative_to(root).as_posix(): p.read_bytes() for p in files}
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
        archive.writestr("UPDATE-MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity failed")
        for name, digest in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise RuntimeError(f"Hash mismatch: {name}")
        # Test the exact payload in a clean install tree using the already
        # installed dependencies, not modules from the working source tree.
        with tempfile.TemporaryDirectory(prefix="rag-update-verify-") as stage:
            archive.extractall(stage)  # Only the allowlisted paths we created.
            result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
                                    cwd=stage, capture_output=True, text=True, timeout=120)
            print(result.stdout)
            print(result.stderr)
            if result.returncode:
                raise RuntimeError(f"Extracted package tests failed; do not install {output}")
    print(json.dumps({"zip": str(output), "files": len(contents) + 1,
                      "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "extracted_package_tests": "passed"}, indent=2))


if __name__ == "__main__":
    main()
