"""Small map acquisition commands; no implicit large downloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .providers import AMapClient, MapServiceError, write_osm_place


def main() -> int:
    parser = argparse.ArgumentParser(prog="vln-maps")
    sub = parser.add_subparsers(dest="command", required=True)
    osm = sub.add_parser("osm-place", help="cache a small OSM snapshot and export a coarse prior")
    osm.add_argument("slug")
    osm.add_argument("name")
    osm.add_argument("--radius-m", type=float, default=400)
    amap = sub.add_parser("amap-geocode", help="check AMap Web Service geocoding")
    amap.add_argument("address")
    amap.add_argument("--city")
    amap.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        if args.command == "osm-place":
            result = write_osm_place(root, args.slug, args.name, args.radius_m)
        else:
            key = args.key_file.read_text().strip() if args.key_file else None
            result = AMapClient(key).geocode(args.address, args.city)
    except (MapServiceError, ValueError, OSError) as exc:
        # AMap errors are sanitized in providers.py; never print a request URL.
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "result": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
