"""Small, auditable OSM and AMap Web Service clients.

The clients never expose a credential through an exception or cache filename.
OSM WGS84 and AMap GCJ02 are intentionally separate types in returned data.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


USER_AGENT = "CrossBuildingFloorNavResearch/0.1 (small cached OSM queries)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
AMAP_BASE = "https://restapi.amap.com"


class MapServiceError(RuntimeError):
    """A remote provider failed; credentials and raw URLs are suppressed."""


class MissingAMapKey(MapServiceError):
    pass


def _json_request(url: str, params: dict[str, str], *, method: str = "GET", timeout: int = 60) -> dict | list:
    encoded = urlencode(params).encode("utf-8")
    request = Request(
        url if method == "POST" else f"{url}?{encoded.decode('ascii')}",
        data=encoded if method == "POST" else None,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        raise MapServiceError(f"map provider HTTP {exc.code}") from None
    except (URLError, TimeoutError, ValueError) as exc:
        raise MapServiceError(f"map provider request failed: {type(exc).__name__}") from None


def osm_search(name: str) -> dict:
    """One human-triggered Nominatim lookup; caller must cache the result."""
    matches = _json_request(NOMINATIM_URL, {"q": name, "format": "jsonv2", "limit": "3", "addressdetails": "0"})
    if not isinstance(matches, list) or not matches:
        raise MapServiceError("OSM place not found")
    return matches[0]


def _bounded_bbox(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    if not (50 <= radius_m <= 750):
        raise ValueError("radius_m must be in [50, 750] for a public Overpass query")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("invalid WGS84 coordinate")
    lat_delta = radius_m / 111_320
    lon_delta = radius_m / max(111_320 * math.cos(math.radians(lat)), 1)
    return lat - lat_delta, lon - lon_delta, lat + lat_delta, lon + lon_delta


def osm_overpass_snapshot(bbox: tuple[float, float, float, float]) -> dict:
    s, w, n, e = bbox
    query = (
        f"[out:json][timeout:60];("
        f"way[building]({s:.7f},{w:.7f},{n:.7f},{e:.7f});"
        f"relation[building]({s:.7f},{w:.7f},{n:.7f},{e:.7f});"
        f"way[highway~'^(footway|pedestrian|path|steps|living_street)$']({s:.7f},{w:.7f},{n:.7f},{e:.7f});"
        f"node[entrance]({s:.7f},{w:.7f},{n:.7f},{e:.7f});"
        f");out body geom;"
    )
    result = _json_request(OVERPASS_URL, {"data": query}, method="POST", timeout=85)
    if not isinstance(result, dict) or "elements" not in result:
        raise MapServiceError("Overpass response has no elements")
    return result


def coarse_osm_prior(snapshot: dict) -> dict:
    """Whitelisted view for the agent; exact entrances/indoor tags stay hidden."""
    retained: list[dict] = []
    for element in snapshot.get("elements", []):
        tags = element.get("tags", {})
        kind = element.get("type")
        if kind not in {"way", "relation"}:
            continue
        if "building" in tags:
            allowed_tags = {key: tags[key] for key in ("building", "name", "name:en", "name:zh", "ref") if key in tags}
        elif (kind == "way" and tags.get("highway") in {"footway", "pedestrian", "path", "steps", "living_street"}
              and tags.get("indoor") not in {"yes", "corridor", "room", "area"}
              and "level" not in tags and "repeat_on" not in tags):
            allowed_tags = {key: tags[key] for key in ("highway", "foot", "access", "surface") if key in tags}
        else:
            continue
        retained.append({key: element[key] for key in ("type", "id", "geometry", "members") if key in element} | {"tags": allowed_tags})
    return {"provider": "OSM", "coordinate_system": "WGS84", "elements": retained}


def write_osm_place(root: Path, slug: str, name: str, radius_m: float = 400) -> dict:
    if not slug or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for ch in slug):
        raise ValueError("slug must contain only lowercase letters, digits, '-' or '_'")
    directory = root / "data" / "osm" / slug
    directory.mkdir(parents=True, exist_ok=True)
    search_file = directory / "search.json"
    if search_file.exists():
        place = json.loads(search_file.read_text())
    else:
        place = osm_search(name)
        search_file.write_text(json.dumps(place, ensure_ascii=False, indent=2) + "\n")
        # Public Nominatim requires low traffic. This one-off pause is only
        # between a successful new search and the next provider request.
        time.sleep(1)
    lat, lon = float(place["lat"]), float(place["lon"])
    bbox = _bounded_bbox(lat, lon, radius_m)
    raw_file = directory / "raw.json"
    if raw_file.exists():
        raw = json.loads(raw_file.read_text())
    else:
        raw = osm_overpass_snapshot(bbox)
        raw_file.write_text(json.dumps(raw, ensure_ascii=False) + "\n")
    prior = coarse_osm_prior(raw)
    prior_file = directory / "prior.json"
    prior_file.write_text(json.dumps(prior, ensure_ascii=False) + "\n")
    metadata = {
        "query": name,
        "resolved_name": place.get("display_name"),
        "osm_type": place.get("osm_type"),
        "osm_id": place.get("osm_id"),
        "centroid_wgs84": {"lat": lat, "lon": lon},
        "bbox_s_w_n_e": bbox,
        "raw_sha256": hashlib.sha256(raw_file.read_bytes()).hexdigest(),
        "prior_sha256": hashlib.sha256(prior_file.read_bytes()).hexdigest(),
        "raw_element_count": len(raw.get("elements", [])),
        "prior_element_count": len(prior["elements"]),
        "attribution": "© OpenStreetMap contributors, ODbL",
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


class AMapClient:
    def __init__(self, key: str | None = None):
        self._key = (key or os.environ.get("AMAP_WEB_KEY", "")).strip()
        if not self._key:
            raise MissingAMapKey("MISSING_AMAP_KEY: configure an AMap Web Service key")

    def _get(self, path: str, params: dict[str, str]) -> dict:
        result = _json_request(f"{AMAP_BASE}{path}", {"key": self._key, **params})
        if not isinstance(result, dict):
            raise MapServiceError("AMap response is not an object")
        if result.get("status") != "1":
            raise MapServiceError(f"AMap error: {result.get('infocode', 'unknown')}")
        return result

    def geocode(self, address: str, city: str | None = None) -> dict:
        params = {"address": address, "output": "JSON"}
        if city:
            params["city"] = city
        return {"provider": "AMap", "coordinate_system": "GCJ02", "response": self._get("/v3/geocode/geo", params)}

    def wgs84_to_gcj02(self, lon: float, lat: float) -> dict:
        point = f"{lon:.6f},{lat:.6f}"
        return {"provider": "AMap", "source_crs": "WGS84", "coordinate_system": "GCJ02", "response": self._get("/v3/assistant/coordinate/convert", {"locations": point, "coordsys": "gps", "output": "JSON"})}

    def walking_route_gcj02(self, origin: tuple[float, float], destination: tuple[float, float]) -> dict:
        fmt = lambda point: f"{point[0]:.6f},{point[1]:.6f}"
        return {"provider": "AMap", "coordinate_system": "GCJ02", "response": self._get("/v5/direction/walking", {"origin": fmt(origin), "destination": fmt(destination), "show_fields": "cost,polyline"})}
