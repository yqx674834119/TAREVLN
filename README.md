# Cross-Building and Cross-Floor Goal Navigation

This is the independent project workspace for A 5F → B 3F navigation. The project is stored under `~/VLN/cross-building-floor-nav` on the Mac and Host 5090. The TARE_WheelTECH production checkout and its separate `sim2real` Git repository are not runtime dependencies of this bootstrap.

## Current stages

- **Map component:** OSM Nominatim resolves a named place, and Overpass downloads a small source snapshot. A strict exporter gives the policy only building identity/outlines and outdoor walkways. Raw OSM stays evaluator-side. AMap Web Service adapter supports geocoding, WGS84→GCJ02 conversion, and walking routes once `AMAP_WEB_KEY` is supplied. AMap coordinates remain tagged `GCJ02`; they are never silently merged with OSM WGS84 geometry.
- **Simulation component:** Isaac Sim 5.1 is the selected high-fidelity target because its published driver floor is compatible with the 580 series on Host 5090 and it includes a Go2 asset. Installation is staged in an isolated Python 3.11 environment. A minimal scene and the robot/skill bridge still require runtime validation.
- **GPU compatibility:** `scripts/setup_gpu_compat.sh` copies the already cached 580.173.02 userspace libraries into this project. `scripts/with_gpu_compat.sh` changes library paths for a child process only; it does not replace system libraries or reboot the host.

## OSM snapshots

```bash
PYTHONPATH=src python3 -m vln_maps.cli osm-place hkust-gz "Hong Kong University of Science and Technology (Guangzhou)" --radius-m 400
PYTHONPATH=src python3 -m vln_maps.cli osm-place harbour-city "Harbour City, Hong Kong" --radius-m 400
```

The output is `data/osm/<slug>/raw.json`, `prior.json`, and `metadata.json`. Review the resolved centroid and building identities before defining A/B; a search match alone is not a verified campus boundary. Data attribution: © OpenStreetMap contributors, ODbL. Public Nominatim is used at most once per place; cached results are reused. The Overpass request is spatially bounded and cached. Bulk data acquisition should use a regional extract instead of public services.

## AMap adapter

Apply for a **Web 服务 API** key. Put it in a `0600` file under `~/VLN/secrets/amap-web-key` on Host 5090, or export `AMAP_WEB_KEY` in a private shell. Do not commit or print it. The CLI takes the file via `--key-file` and returns a structured `MISSING_AMAP_KEY` status if absent. Example after provisioning:

```bash
PYTHONPATH=src python3 -m vln_maps.cli amap-geocode '香港科技大学（广州）' --key-file ~/VLN/secrets/amap-web-key
```

`walking_route` consumes **GCJ02** endpoints. Convert WGS84 points with the AMap conversion endpoint before calling it, and keep each provider's coordinates labeled. Hong Kong coverage and the account's regional entitlement require a real API response before either is claimed.

## Host 5090 installation

The bootstrap is launched in the background, with a PID and a full log under `~/VLN/logs/`. It installs no system packages and does not start TARE, a real robot, or a network-facing service.

```bash
nohup bash ~/VLN/cross-building-floor-nav/scripts/install_isaac51.sh \
  > ~/VLN/logs/isaac51-install.log 2>&1 < /dev/null &
echo $! > ~/VLN/logs/isaac51-install.pid
```

The runtime script will not pre-accept NVIDIA's EULA. A human must accept it before the first Isaac Sim launch. Installation completion, GPU graphics initialization, Go2 articulation, elevator behavior, ROS 2 sensor timing, and end-to-end navigation are separate acceptance gates.

