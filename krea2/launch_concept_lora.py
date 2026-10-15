"""launch_concept_lora.py — spawn the 4 artist concept-LoRA trainers.

One tmux session per artist, one GPU each (measured 35.5 GB peak at
1024px, so any free 96 GB card works; GPU3's 6 GB viz server is fine).
Per-artist configs are materialized from the template config into the
artist's run dir (the pilot's identity-check discipline: an existing
session name refuses a second launch).

Usage:
    uv run python krea2/launch_concept_lora.py                 # launch all 4
    uv run python krea2/launch_concept_lora.py --status        # summarize
    uv run python krea2/launch_concept_lora.py --render        # render grids
    uv run python krea2/launch_concept_lora.py --only chunie --gpu 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "krea2", "configs", "train_concept_lora.json")
RUN_ROOT = os.path.join(ROOT, "runs", "k2-concept-lora")

# (artist, gpu) — one 96 GB card each; GPU3 shares with the 6 GB viz server.
DEFAULT_ASSIGNMENT = [
    ("chunie", 0),
    ("darkgem", 1),
    ("dangpa", 2),
    ("meesh", 3),
]


def _artist_cfg(template: dict, artist: str, gpu: int) -> dict:
    cfg = json.loads(json.dumps(template))  # deep copy
    cfg["artist"] = artist
    cfg["trigger"] = f"by {artist}"
    cfg["device"] = f"cuda:{gpu}"
    run_dir = os.path.join(RUN_ROOT, artist)
    cfg["ckpt_path"] = os.path.join(run_dir, "ckpts")
    cfg["preview_path"] = os.path.join(run_dir, "previews")
    cfg["cache_path"] = os.path.join(run_dir, "cache")
    return cfg


def _tmux_sessions() -> set[str]:
    out = subprocess.run(["tmux", "ls"], capture_output=True, text=True)
    return {line.split(":")[0] for line in out.stdout.splitlines() if line.strip()}


def launch(template: dict, only: list[str] | None = None,
           gpu_map: dict[str, int] | None = None, dry: bool = False) -> None:
    gpu_map = gpu_map or {}
    busy = _tmux_sessions()
    for artist, default_gpu in DEFAULT_ASSIGNMENT:
        if only and artist not in only:
            continue
        gpu = gpu_map.get(artist, default_gpu)
        session = f"k2-concept-{artist}"
        if session in busy:
            print(f"[skip] tmux session '{session}' already exists.")
            continue
        cfg = _artist_cfg(template, artist, gpu)
        run_dir = os.path.join(RUN_ROOT, artist)
        os.makedirs(run_dir, exist_ok=True)
        cfg_path = os.path.join(run_dir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        log_path = os.path.join(run_dir, "train.log")
        cmd = (f"uv run python krea2/train_concept_lora.py {cfg_path} "
               f">> {log_path} 2>&1")
        if dry:
            print(f"[dry] tmux new-session -d -s {session} '{cmd}'")
            continue
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, cmd],
            check=True, cwd=ROOT,
        )
        print(f"[launch] {artist} on {cfg['device']} -> tmux:{session} "
              f"(log {log_path})")


def status() -> None:
    busy = _tmux_sessions()
    for artist, gpu in DEFAULT_ASSIGNMENT:
        session = f"k2-concept-{artist}"
        run_dir = os.path.join(RUN_ROOT, artist)
        result = os.path.join(run_dir, "ckpts", "result.json")
        ckpt = os.path.join(run_dir, "ckpts",
                            "concept_step_%d_final.safetensors"
                            % json.load(open(TEMPLATE)).get("max_steps", 400))
        if session in busy:
            print(f"[{artist}] RUNNING on GPU{gpu} (tmux {session}).")
        elif os.path.exists(result):
            with open(result) as f:
                data = json.load(f)
            print(f"[{artist}] DONE: {data['global_step']} steps, "
                  f"{data['seconds_per_step']:.2f}s/step, "
                  f"peak {data.get('peak_allocated_gib', 0):.1f} GiB.")
        elif os.path.exists(ckpt):
            print(f"[{artist}] finished checkpoint present (no tmux).")
        else:
            log = os.path.join(run_dir, "train.log")
            tail = ""
            if os.path.exists(log):
                with open(log, "rb") as f:
                    f.seek(max(0, os.path.getsize(log) - 400))
                    tail = f.read().decode(errors="replace").replace("\n", " | ")[-200:]
            print(f"[{artist}] NOT RUNNING. tail: {tail or '(no log)'}")


def _render_one(artist: str, gpu: int, template: dict) -> None:
    """One artist's 8-prompt render + grid (runs as its own process/GPU)."""
    from PIL import Image

    prompts = [
        "A fully clothed anthropomorphic fox explorer wearing a green jacket, standing beside a mountain lake at sunrise.",
        "A cheerful anthropomorphic rabbit in a yellow raincoat holding an umbrella on a rainy city street.",
        "An anthropomorphic wolf musician in a formal suit playing a violin in a warmly lit concert hall.",
        "An anthropomorphic cat librarian in a sweater reading a book by a window, autumn leaves outside.",
        "A red bicycle leaning against a white garden wall, flowering vines, afternoon sunlight.",
        "A cozy wooden cabin beside a snowy forest, warm light in the windows, twilight.",
        "A bowl of oranges on a blue tablecloth beside a ceramic teapot, still life.",
        "A sailing boat in a quiet harbor with colorful waterfront buildings at dawn.",
    ]
    cfg = _artist_cfg(template, artist, gpu)
    run_dir = os.path.join(RUN_ROOT, artist)
    ckpt = os.path.join(run_dir, "ckpts",
                        f"concept_step_{cfg['max_steps']}_final.safetensors")
    if not os.path.exists(ckpt):
        print(f"[render] {artist}: no final checkpoint at {ckpt}; skipping.")
        return
    out_dir = os.path.join(run_dir, "renders")
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "render_prompts.json")
    manifest = [
        dict(prompt=f"<concept:{artist}> {p}", seed=4000 + i)
        for i, p in enumerate(prompts)
    ]
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    tag_cfg = {
        "enabled": True,
        "vocab_path": os.path.join(run_dir, "ckpts", "concept_vocab.parquet"),
        "max_tags": 1,
        "tag_drop_prob": 0.0,
    }
    infer_cfg_path = os.path.join(run_dir, "infer_config.json")
    infer_cfg = {
        "mmdit_config": template.get("mmdit_config", "large_wide"),
        "encoder_config": template.get("encoder_config", "qwen3_vl_4b"),
        "ae_model_id": template.get("ae_model_id", "Qwen/Qwen-Image"),
        "encoder_model_id": template.get(
            "encoder_model_id", "Qwen/Qwen3-VL-4B-Instruct"),
        "lora_rank": template.get("lora_rank", 16),
        "lora_alpha": template.get("lora_alpha", 16.0),
        "tag_embed": tag_cfg,
    }
    with open(infer_cfg_path, "w") as f:
        json.dump(infer_cfg, f, indent=2)
    cmd = [
        "uv", "run", "python", os.path.join(ROOT, "krea2", "inference.py"),
        "--config", infer_cfg_path,
        "--mmdit-checkpoint", template["mmdit_checkpoint"],
        "--device", f"cuda:{gpu}", "--batch-size", "1",
        "--steps", "28", "--guidance", "4.5", "--mu", "1.15",
        "--width", "1024", "--height", "1024",
        "--prompts-file", manifest_path,
        "--out-dir", out_dir,
        "--parse-concept-markers",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    # Grid.
    rows = json.load(open(os.path.join(out_dir, "manifest.json")))
    thumbs = []
    for row, expected in zip(rows, manifest):
        assert row["prompt"] == expected["prompt"]
        with Image.open(os.path.join(out_dir, row["image"])) as im:
            thumbs.append(im.convert("RGB").resize((256, 256)))
    grid = Image.new("RGB", (4 * 256, 2 * 256))
    for i, im in enumerate(thumbs):
        grid.paste(im, ((i % 4) * 256, (i // 4) * 256))
    grid_path = os.path.join(out_dir, "grid.png")
    grid.save(grid_path)
    print(f"[render] {artist}: grid -> {grid_path}", flush=True)


def render(template: dict, only: list[str] | None = None,
           gpu_map: dict[str, int] | None = None) -> None:
    """Post-training render, one process PER ARTIST on its own GPU.

    Every artist renders on the GPU its trainer used, so all four run
    concurrently — the wall clock is one artist's render, not the sum.
    """
    from concurrent.futures import ThreadPoolExecutor

    gpu_map = gpu_map or {}
    jobs = [(a, gpu_map.get(a, g)) for a, g in DEFAULT_ASSIGNMENT
            if not only or a in only]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futs = {pool.submit(_render_one, a, g, template): a for a, g in jobs}
        for fut, artist in futs.items():
            try:
                fut.result()
            except subprocess.CalledProcessError as e:
                print(f"[render] {artist}: FAILED (exit {e.returncode}).")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="subset of artists")
    ap.add_argument("--gpu", action="append", default=[],
                    help="artist=GPU override, e.g. --gpu chunie=2")
    ap.add_argument("--dry", action="store_true", help="print tmux commands only")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--render", action="store_true",
                    help="render final checkpoints instead of launching")
    args = ap.parse_args()

    with open(TEMPLATE) as f:
        template = json.load(f)
    gpu_map = {}
    for pair in args.gpu:
        name, _, num = pair.partition("=")
        gpu_map[name.strip()] = int(num)

    if args.status:
        status()
    elif args.render:
        render(template, only=args.only, gpu_map=gpu_map)
    else:
        launch(template, only=args.only, gpu_map=gpu_map, dry=args.dry)


if __name__ == "__main__":
    main()
