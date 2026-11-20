"""Stage or archive the RHEL bundle, excluding workstation state and credentials.

python deploy/rhel/package.py stage OUTPUT_DIRECTORY
python deploy/rhel/package.py archive OUTPUT_DIRECTORY
"""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
action, target = sys.argv[1], Path(sys.argv[2]).resolve()

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

if action == "stage":
    target.mkdir(parents=True, exist_ok=False)
    for name in ("app", "tests", "deploy/rhel"):
        for source in (root / name).rglob("*"):
            if source.is_file() and (source.suffix in {".py", ".html", ".css", ".js", ".sh", ".md", ".example"} or source.name == "Dockerfile.builder"):
                dest = target / source.relative_to(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    for name in ("requirements.txt", "requirements-native.txt", "requirements-gpu.txt", "README.md", "LICENSE", ".env.example"):
        shutil.copy2(root / name, target / name)
    for name in ("models--nomic-ai--nomic-embed-text-v1.5", "models--Xenova--ms-marco-MiniLM-L-6-v2"):
        shutil.copytree(root / ".data/models" / name, target / ".data/models" / name, symlinks=False)
        # FastEmbed's metadata keys use host path separators. Make the staged
        # copy portable to Linux while leaving model contents untouched.
        metadata = target / ".data/models" / name / "files_metadata.json"
        if metadata.exists():
            entries = json.loads(metadata.read_text(encoding="utf-8"))
            metadata.write_text(json.dumps({k.replace("\\", "/"): v for k, v in entries.items()}), encoding="utf-8")
elif action in {"manifest", "archive"}:
    files = sorted(p for p in target.rglob("*") if p.is_file() and
                   not any(x in {".venv-rhel", "__pycache__"} for x in p.relative_to(target).parts) and
                   p.name not in {"SHA256SUMS", ".env"} and
                   (p.relative_to(target).parts[0] != ".data" or p.relative_to(target).parts[:2] == (".data", "models")))
    manifest = target / "SHA256SUMS"
    manifest.write_text("".join(f"{digest(p)}  {p.relative_to(target).as_posix()}\n" for p in files), encoding="utf-8", newline="\n")
    if action == "manifest":
        print(f"Manifest: {len(files)} files")
        raise SystemExit(0)
    archive = target.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for p in files + [manifest]:
            z.write(p, f"{target.name}/{p.relative_to(target).as_posix()}")
    archive.with_suffix(".zip.sha256").write_text(f"{digest(archive)}  {archive.name}\n", encoding="utf-8")
    print(f"{archive}: {archive.stat().st_size:,} bytes")
else:
    raise SystemExit("Expected stage, manifest or archive")
