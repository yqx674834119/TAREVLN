"""P1 diagnostic frontend: decode only rendered RGB and project using measured depth.

QR/text signs replace OCR/VLM for the initial integration baseline. No USD prim,
oracle, sign placement list, hidden floor id, or semantic segmentation is read.
"""
from __future__ import annotations

import hashlib
import json
import math

import zxingcpp
import numpy as np


class SignFrontend:
    def __init__(self):
        pass

    def decode(self, rgb, depth, intrinsics, camera_position, camera_rotation):
        if rgb is None or depth is None or rgb.size == 0:
            return []
        image = np.ascontiguousarray(rgb[:, :, :3])
        codes = zxingcpp.read_barcodes(image, formats=zxingcpp.BarcodeFormat.QRCode)
        if not codes:
            return []
        evidence = "rgb-sha256:" + hashlib.sha256(image.tobytes()).hexdigest()
        result = []
        for code in codes:
            corners = code.position
            vertices = np.array([[p.x, p.y] for p in (corners.top_left, corners.top_right, corners.bottom_right, corners.bottom_left)])
            try:
                payload = json.loads(code.text)
            except (ValueError, TypeError):
                continue
            if not isinstance(payload, dict) or payload.get("kind") not in {"lift", "cabin", "floor", "building", "exit", "bridge"}:
                continue
            u, v = vertices.mean(axis=0)
            i, j = int(round(v)), int(round(u))
            if not (2 <= i < depth.shape[0] - 2 and 2 <= j < depth.shape[1] - 2):
                continue
            patch = depth[i-2:i+3, j-2:j+3]
            valid = patch[np.isfinite(patch) & (patch > .1) & (patch < 20)]
            if len(valid) < 5:
                continue
            d = float(np.median(valid))
            fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
            ray_camera = np.array([d, (cx - u) * d / fx, (cy - v) * d / fy])
            world = np.asarray(camera_position) + np.asarray(camera_rotation) @ ray_camera
            result.append({"text": payload, "point": world.tolist(), "pixel": [float(u), float(v)],
                           "range_m": d, "evidence": evidence})
        return result
