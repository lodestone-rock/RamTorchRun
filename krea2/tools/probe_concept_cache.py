"""probe_concept_cache.py — time the two cache phases (VAE vs text encoder).

Runs the trainer's build_cache on a THROWAWAY cache dir and timestamps every
safetensors write by which subdirectory it lands in (`latents/` = VAE phase,
`text/` = encoder phase). Reports per-item and per-phase totals, then deletes
the throwaway dir. The real cache is untouched (fingerprint-reused).

Usage: uv run python krea2/tools/probe_concept_cache.py [config.json] [--keep]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from safetensors.torch import save_file as _orig_save_file

events: list[tuple[float, str]] = []
_t0 = time.perf_counter()


def _timed_save_file(tensor_data, path, **kw):
    events.append((time.perf_counter() - _t0, path))
    return _orig_save_file(tensor_data, path, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?",
                    default="runs/k2-concept-lora/chunie/config.json")
    ap.add_argument("--keep", action="store_true",
                    help="keep the throwaway cache dir (prints its path)")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    scratch = tempfile.mkdtemp(prefix="concept_cache_probe_")
    cfg = dict(cfg)
    cfg["cache_path"] = os.path.join(scratch, "cache")
    cfg["device"] = cfg.get("device", "cuda:0")

    import safetensors.torch
    safetensors.torch.save_file = _timed_save_file
    # trainer imports save_file by name at module import; patch it there too
    import krea2.train_concept_lora as tcl
    tcl.save_file = _timed_save_file

    t_start = time.perf_counter()
    manifest = tcl.build_cache(cfg)
    total = time.perf_counter() - t_start

    lat = [(t, p) for t, p in events if "/latents/" in p]
    txt = [(t, p) for t, p in events if "/text/" in p]

    def phase(name, ev, n_items):
        if not ev:
            print(f"{name}: no writes recorded")
            return
        t_first_file = ev[0][0]
        t_last_file = ev[-1][0]
        span = t_last_file - t_first_file
        print(f"\n{name}: {len(ev)} writes over {span:.1f}s "
              f"({span / max(len(ev), 1):.2f} s/item, n={n_items})")
        print(f"  first write at +{t_first_file:.1f}s (incl. model load), "
              f"last at +{t_last_file:.1f}s")

    n_lat = len(manifest["images"])
    n_txt = len({os.path.basename(i["bands"][b]["path"])
                 for i in manifest["images"] for b in i["bands"]}) + 1

    print(f"[probe] artist={cfg['artist']} cache={cfg['cache_path']}")
    phase("VAE  (latents/)", lat, n_lat)
    phase("TEXT (text/)   ", txt, n_txt)
    print(f"\nTOTAL wall (build_cache): {total:.1f}s")
    print(f"  VAE   share: {(lat[-1][0] - lat[0][0]) / total * 100 if lat else 0:.0f}%")
    print(f"  TEXT  share: {(txt[-1][0] - txt[0][0]) / total * 100 if txt else 0:.0f}%")

    if args.keep:
        print(f"[probe] kept scratch cache: {scratch}")
    else:
        shutil.rmtree(scratch, ignore_errors=True)
        print(f"[probe] removed scratch {scratch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())