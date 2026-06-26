"""Content-addressed artifact store (parquet + npy).

Activations and explanations are expensive — cache aggressively and key caches
by a hash of ``(model_rev, nla_rev, config)`` so reruns are resumable and a
config change cannot silently reuse stale artifacts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def config_hash(config: dict[str, Any], *, length: int = 16) -> str:
    """Stable short hash of a JSON-serializable config dict."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:length]


def _to_record(obj: Any) -> dict[str, Any]:
    if is_dataclass(obj) and not isinstance(obj, type):
        out = {}
        for k, v in asdict(obj).items():
            # enums -> their value; tuples -> lists for parquet friendliness
            if hasattr(v, "value"):
                out[k] = v.value
            elif isinstance(v, tuple):
                out[k] = list(v)
            else:
                out[k] = v
        return out
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"cannot convert {type(obj)!r} to a record")


class ContentStore:
    """A directory-backed store, namespaced by a config hash."""

    def __init__(self, root: str | Path, config: dict[str, Any] | None = None):
        self.root = Path(root)
        self.namespace = config_hash(config) if config else "default"
        self.base = self.root / self.namespace
        self.base.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str, suffix: str) -> Path:
        safe = hashlib.sha1(key.encode()).hexdigest()[:24]
        return self.base / f"{safe}{suffix}"

    # ─── arrays ──────────────────────────────────────────────────────────
    def has_array(self, key: str) -> bool:
        return self.path_for(key, ".npy").exists()

    def put_array(self, key: str, arr: np.ndarray) -> str:
        p = self.path_for(key, ".npy")
        np.save(p, np.asarray(arr))
        return str(p)

    def get_array(self, key: str) -> np.ndarray:
        p = self.path_for(key, ".npy")
        assert p.exists(), f"no cached array for key {key!r}"
        return np.load(p)

    # ─── record tables ───────────────────────────────────────────────────
    def put_records(self, name: str, records: Sequence[Any]) -> str:
        import pandas as pd

        df = pd.DataFrame([_to_record(r) for r in records])
        p = self.base / f"{name}.parquet"
        df.to_parquet(p, index=False)
        return str(p)

    def get_records(self, name: str) -> "Any":
        import pandas as pd

        p = self.base / f"{name}.parquet"
        assert p.exists(), f"no cached table {name!r}"
        return pd.read_parquet(p)

    def has_records(self, name: str) -> bool:
        return (self.base / f"{name}.parquet").exists()
