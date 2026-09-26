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
    convert = sub.add_parser("amap-convert", help="convert a WGS84 point to GCJ02 through AMap")
    convert.add_argument("lon", type=float)
    convert.add_argument("lat", type=float)
    walk = sub.add_parser("amap-walk", help="request an AMap walking route between GCJ02 points")
    walk.add_argument("origin_lon", type=float)
    walk.add_argument("origin_lat", type=float)
    walk.add_argument("destination_lon", type=float)
    walk.add_argument("destination_lat", type=float)
    for command in (amap, convert, walk):
        command.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        if args.command == "osm-place":
            result = write_osm_place(root, args.slug, args.name, args.radius_m)
        elif args.command.startswith("amap-"):
            key = args.key_file.read_text().strip() if args.key_file else None
            client = AMapClient(key)
            if args.command == "amap-geocode":
                result = client.geocode(args.address, args.city)
            elif args.command == "amap-convert":
                result = client.wgs84_to_gcj02(args.lon, args.lat)
            else:
                result = client.walking_route_gcj02(
                    (args.origin_lon, args.origin_lat),
                    (args.destination_lon, args.destination_lat),
                )
    except (MapServiceError, ValueError, OSError) as exc:
        # AMap errors are sanitized in providers.py; never print a request URL.
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "result": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
