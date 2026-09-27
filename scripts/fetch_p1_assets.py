"""Download the published Go2 policy/model and reusable-scene root manifests.

Run with nohup on 5090. Scene root files are inventories, not complete offline
scene packages: their nested USD/texture/MDL dependencies must also be resolved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

BASE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/"
FILES = {
    "go2/checkpoint.pt": "IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt",
    "go2/go2_lab.usd": "IsaacLab/Robots/Unitree/Go2/go2.usd",
    "go2/Props/instanceable_meshes.usd": "IsaacLab/Robots/Unitree/Go2/Props/instanceable_meshes.usd",
    "scene_manifests/hospital.usd": "Environments/Hospital/hospital.usd",
    "scene_manifests/office.usd": "Environments/Office/office.usd",
    "scene_manifests/simple_room.usd": "Environments/Simple_Room/simple_room.usd",
}
EXPECTED_SHA256 = {
    "go2/checkpoint.pt": "47ccb2046ec3064257997f879c62d64217b937713c30d827883b2c2f40c50439",
    "go2/go2_lab.usd": "ba171c972b987d8c8fb7157ccad2ba9c0c1fed105755d2d8af46bef96cc11c6d",
    "go2/Props/instanceable_meshes.usd": "2902646d0f4c13c9ecae3ac9046e32c7d18902eef9679de9829f642a15c93fb5",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.home() / "VLN/assets")
    args = parser.parse_args()
    records = []
    for relative, source in FILES.items():
        path = args.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"path": str(path), "url": BASE + source}
        try:
            if not path.is_file():
                temporary = path.with_suffix(path.suffix + ".partial")
                with urllib.request.urlopen(record["url"], timeout=90) as response, temporary.open("wb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                temporary.replace(path)
            record.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), status="downloaded")
            if relative in EXPECTED_SHA256 and record["sha256"] != EXPECTED_SHA256[relative]:
                raise ValueError("Downloaded Go2 asset does not match the validated SHA-256")
        except Exception as error:
            record.update(status="error", error=str(error))
        records.append(record)
        print(json.dumps(record), flush=True)
        (args.root / "p1_manifest.json").write_text(json.dumps(records, indent=2))
    if any(r["status"] != "downloaded" for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
