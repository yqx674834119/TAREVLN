"""Image-capable DeepSeek adapter with private credentials and evidence logs."""
from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


class VisionAPI:
    def __init__(self, output: Path, key_file: Path | None = None, max_calls=60):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        key_file = key_file or Path.home() / "VLN/secrets/deepseek-api-key"
        if key_file.stat().st_mode & 0o077:
            raise PermissionError("VLM key file must have mode 0600")
        self._key = key_file.read_text().strip()
        self.model = "deepseek-flash"
        self.calls, self.max_calls = 0, max_calls

    def ask(self, prompt: str, images: list[bytes], context: dict | None = None):
        if self.calls >= self.max_calls:
            raise RuntimeError("VLM call budget exhausted; stop the task")
        self.calls += 1
        call_dir = self.output / f"call-{self.calls:04d}"
        call_dir.mkdir(exist_ok=False)
        parts = [{"type": "text", "text": prompt}]
        hashes = []
        for index, img in enumerate(images):
            (call_dir / f"image-{index}.jpg").write_bytes(img)
            hashes.append(hashlib.sha256(img).hexdigest())
            parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(img).decode()}})
        body = {"model": self.model, "messages": [{"role": "user", "content": parts}],
                "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"},
                "max_tokens": 1000, "temperature": 0}
        record = {"model": self.model, "prompt": prompt, "image_sha256": hashes, "context": context or {}, "started": time.time()}
        # Store neither request headers nor image data URLs in the JSON log.
        (call_dir / "request.json").write_text(json.dumps(record, indent=2))
        request = urllib.request.Request("https://api.deepseek.com/chat/completions",
                                         data=json.dumps(body).encode(), method="POST",
                                         headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
            content = payload["choices"][0]["message"]["content"]
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ValueError("VLM result must be a JSON object")
            record.update(status="ok", result=result, usage=payload.get("usage"),
                          response_id=payload.get("id"), latency_seconds=time.time() - record["started"])
            (call_dir / "response.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
            return result
        except urllib.error.HTTPError as error:
            record.update(status="error", error=f"HTTP {error.code}")
            (call_dir / "response.json").write_text(json.dumps(record, indent=2))
            raise RuntimeError(f"VLM request failed with HTTP {error.code}") from None
        except Exception as error:
            record.update(status="error", error=type(error).__name__)
            (call_dir / "response.json").write_text(json.dumps(record, indent=2))
            raise
