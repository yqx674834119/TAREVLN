"""Recursively cache an official USD scene, including materials and textures.

Runs outside Kit with usd-core. Writes an atomic inventory; a root USD alone is
never reported as a complete scene. Credentials are not used by this downloader.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request
from pxr import Sdf, UsdUtils

BASE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="Isaac/Environments/Office/office.usd")
    ap.add_argument("--root", type=Path, default=Path.home() / "VLN/assets/usd-cache")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if not 1 <= args.workers <= 32:
        ap.error("workers must be between 1 and 32")
    args.root.mkdir(parents=True, exist_ok=True)
    pending, seen, records = {urllib.parse.urljoin(BASE, args.scene)}, set(), []

    def fetch(url):
        if not url.startswith(BASE):
            return {"url": url, "status": "unsupported_external"}, []
        relative = urllib.parse.unquote(url[len(BASE):])
        path = (args.root / relative).resolve()
        if not path.is_relative_to(args.root.resolve()):
            raise ValueError("Asset path escapes cache")
        record = {"url": url, "path": str(path)}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                for attempt in range(4):
                    try:
                        part = path.with_suffix(path.suffix + ".part")
                        offset = part.stat().st_size if part.exists() else 0
                        req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
                        stream = urllib.request.urlopen(req, timeout=90)
                        mode = "ab" if offset and stream.status == 206 else "wb"
                        with stream, part.open(mode) as out:
                            while chunk := stream.read(1024 * 1024):
                                out.write(chunk)
                        path.with_suffix(path.suffix + ".part").replace(path)
                        break
                    except Exception:
                        if attempt == 3:
                            raise
                        time.sleep(2 ** attempt)
            deps = set()
            if path.suffix.lower() in {".usd", ".usda", ".usdc"}:
                layer = Sdf.Layer.FindOrOpen(str(path))
                if not layer:
                    raise ValueError("USD layer could not be opened")
                assets = re.findall(r"@([^@]+)@", layer.ExportToString())
                def rewrite(asset):
                    if not asset or (asset.endswith(".mdl") and "/" not in asset):
                        return asset  # Kit built-in MDL module, e.g. OmniPBR.mdl
                    source = urllib.parse.urljoin(url, asset)
                    deps.add(source)
                    if source.startswith(BASE):
                        dest = args.root / urllib.parse.unquote(source[len(BASE):])
                        return os.path.relpath(dest, path.parent)
                    return asset
                for asset in assets:
                    rewrite(asset)
                UsdUtils.ModifyAssetPaths(layer, rewrite)
                layer.Save()
            elif path.suffix.lower() == ".mdl":
                mdl = path.read_text(errors="replace")
                for asset in re.findall(r'(?:texture_2d|texture_3d|texture_cube)\s*\(\s*"([^"]+)"', mdl):
                    deps.add(urllib.parse.urljoin(url, asset))
                modules = re.findall(r'import\s+((?:\.\.?::)[A-Za-z0-9_:]+)::\*', mdl)
                modules += re.findall(r'using\s+((?:\.\.?::)[A-Za-z0-9_:]+)\s+import', mdl)
                for module in modules:
                    relative = module.replace("..::", "../").replace(".::", "./").replace("::", "/") + ".mdl"
                    deps.add(urllib.parse.urljoin(url, relative))
            record.update(status="cached", bytes=path.stat().st_size,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            return record, deps
        except Exception as error:
            return dict(record, status="error", error=str(error)), []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = set()
        while pending or futures:
            batch = sorted(pending - seen, key=lambda u: (0 if u.endswith((".usd", ".usda", ".usdc", ".mdl")) else 1, u))[:args.workers-len(futures)]
            for url in batch:
                seen.add(url)
                futures.add(pool.submit(fetch, url))
            pending.difference_update(seen)
            done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                record, deps = future.result()
                records.append(record)
                pending.update(set(deps) - seen)
                print(json.dumps(record), flush=True)
            manifest = {"scene": args.scene, "complete": not pending and not futures and all(r["status"] == "cached" for r in records), "pending": len(pending)+len(futures), "files": records}
            dest = args.root / "office-download.json"
            dest.with_suffix(".tmp").write_text(json.dumps(manifest, indent=2))
            dest.with_suffix(".tmp").replace(dest)
    if not manifest["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
