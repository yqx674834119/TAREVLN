# Cross-Building and Cross-Floor Goal Navigation

This is the independent project workspace for A 5F → B 3F navigation. The project is stored under `~/VLN/cross-building-floor-nav` on the Mac and Host 5090. The TARE_WheelTECH production checkout and its separate `sim2real` Git repository are not runtime dependencies of this bootstrap.

## Current stages

- **Map component:** OSM Nominatim resolves a named place, and Overpass downloads a small source snapshot. A strict exporter gives the policy only building identity/outlines and outdoor walkways. Raw OSM stays evaluator-side. AMap Web Service adapter supports geocoding, WGS84→GCJ02 conversion, and walking routes once `AMAP_WEB_KEY` is supplied. AMap coordinates remain tagged `GCJ02`; they are never silently merged with OSM WGS84 geometry.
- **Task component:** `vln_core.connections` is the initial directed connection-memory contract. A sighting or sign may guide investigation, but only a traversal plus distinct arrival-identity evidence becomes a reusable route. Capability restrictions, wrong signs, blocked edges and direction are represented; the P1 diagnostic now connects visible-sign perception and motor execution, while the proposed research planner remains to be integrated.
- **Simulation component:** Isaac Sim 5.1 is installed and the articulated Go2 has completed the synthetic P1 A5F → A1 → outdoor → B1 → B3F route on Host 5090. The baseline uses rendered QR signs, depth, ideal LiDAR/odometry and an API-assisted physical elevator. This is an integration baseline; TARE/VLM, real-map alignment and real-robot validation remain pending. Non-foot thigh contacts were recorded, so reaching the goal is not a clean locomotion-quality pass. See [P1 runbook](docs/p1-runbook.md) and [audit results](docs/p1-results.md).
- **GPU compatibility:** `scripts/setup_gpu_compat.sh` copies the already cached 580.173.02 userspace libraries into this project. `scripts/with_gpu_compat.sh` changes library paths for a child process only; it does not replace system libraries or reboot the host.

## OSM snapshots

```bash
PYTHONPATH=src python3 -m vln_maps.cli osm-place hkust-gz "Hong Kong University of Science and Technology (Guangzhou)" --radius-m 400
PYTHONPATH=src python3 -m vln_maps.cli osm-place harbour-city "Harbour City, Hong Kong" --radius-m 400
```

The output is `data/osm/<slug>/raw.json`, `prior.json`, and `metadata.json`. Review the resolved centroid and building identities before defining A/B; a search match alone is not a verified campus boundary. Data attribution: © OpenStreetMap contributors, ODbL. Public Nominatim is used at most once per place; cached results are reused. The Overpass request is spatially bounded and cached. Bulk data acquisition should use a regional extract instead of public services.

The current 400 m HKUST(GZ) snapshot has 29 building features and only 4 outdoor walkway features; Harbour City has 231 buildings and 448 walkway features. These are small map-acquisition checks, not validated navigable site graphs. The synthetic A/B simulator is intentionally independent of both real-world snapshots until building identities, floor plans, legal access and connection ground truth are supplied.

## AMap adapter

Apply for a **Web 服务 API** key. Put it in a `0600` file under `~/VLN/secrets/amap-web-key` on Host 5090, or export `AMAP_WEB_KEY` in a private shell. Do not commit or print it. The CLI takes the file via `--key-file` and returns a structured `MISSING_AMAP_KEY` status if absent. The adapter and keyless error path are implemented; live AMap calls cannot be validated until a key is available. Examples after provisioning:

```bash
PYTHONPATH=src python3 -m vln_maps.cli amap-geocode '香港科技大学（广州）' --key-file ~/VLN/secrets/amap-web-key
PYTHONPATH=src python3 -m vln_maps.cli amap-convert 113.4784431 22.8875346 --key-file ~/VLN/secrets/amap-web-key
# Pass GCJ02 coordinates returned by AMap, not raw OSM WGS84:
PYTHONPATH=src python3 -m vln_maps.cli amap-walk ORIGIN_LON ORIGIN_LAT DEST_LON DEST_LAT --key-file ~/VLN/secrets/amap-web-key
```

`walking_route` consumes **GCJ02** endpoints. Convert WGS84 points with the AMap conversion endpoint before calling it, and keep each provider's coordinates labeled. Hong Kong coverage and the account's regional entitlement require a real API response before either is claimed.

## Host 5090 installation

The bootstrap is launched in the background, with a PID and a full log under `~/VLN/logs/`. It installs no system packages and does not start TARE, a real robot, or a network-facing service.

```bash
nohup bash ~/VLN/cross-building-floor-nav/scripts/install_isaac51.sh \
  > ~/VLN/logs/isaac51-install.log 2>&1 < /dev/null &
echo $! > ~/VLN/logs/isaac51-install.pid
```

The user explicitly accepted NVIDIA's EULA for this task. Installation, GPU startup, articulated dynamics, camera/depth/IMU and physical elevator travel have now run. Native simulation, ROS 2 integration, and real-robot validation remain separate acceptance gates.

## Earlier MuJoCo smoke test

On Host 5090, with no reboot or system driver changes:

```bash
cd ~/VLN/cross-building-floor-nav
python3 sim/generate_demo_scene.py
MENAGERIE_CACHE_DIR=~/VLN/cache/mujoco_menagerie MUJOCO_GL=egl \
  scripts/with_gpu_compat.sh ~/VLN/envs/mujoco-go2/bin/python sim/run_mujoco_smoke.py
```

The report and frames are `data/sim/demo/mujoco_smoke.json`, `mujoco_smoke.png` (scene overview), and `mujoco_go2_a5f.png` (robot closeup). This smoke test checks only that the Go2 asset compiles in the campus scene, remains supported at A/F5 for 400 physics steps, and can be rendered through the project-local NVIDIA library path. The static `oracle.json` declares elevator and portal transitions for a future simulator skill API; they are **not** working robot actions. The companion `prior.json` deliberately omits those connections.

The preceding MuJoCo files belong to the earlier static bootstrap. The implemented doors/elevators and closed loop are in `sim/build_p1_usd.py` and `sim/run_p1.py`; use the P1 runbook for current commands. The simulator environment is `~/VLN/envs/isaac51`.

After you read and accept the [NVIDIA Omniverse license](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html), the prepared stage-only check is:

```bash
OMNI_KIT_ACCEPT_EULA=YES scripts/with_gpu_compat.sh \
  ~/VLN/envs/isaac51/bin/python sim/run_isaac_stage_smoke.py
```

This lightweight stage diagnostic is separate from the P1 route test. Downloads can be prepared in the background with `scripts/install_p1_components.sh`; it does not launch Isaac or accept a license automatically.
