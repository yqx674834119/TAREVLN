#!/usr/bin/env bash
set -euo pipefail
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project"
export PYTHONPATH="$project/src"
python3 -m vln_maps.cli osm-place hkust-gz 'Hong Kong University of Science and Technology (Guangzhou)' --radius-m 400
sleep 2
python3 -m vln_maps.cli osm-place harbour-city 'Harbour City, Hong Kong' --radius-m 400
