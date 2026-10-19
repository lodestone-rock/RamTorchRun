"""Single-GPU throughput probe for moss/train_sft.py: per-GPU batch x grad checkpointing x torch.compile.

Real MOSS transformer (fp32 master, bf16 autocast, batched forward), real batched text encoder on real
trainer_samples_v3 captions (the six SFT columns, drawn evenly), real VAE encode of random pixels at the 512
buckets, fused AdamW with its state allocated. Per config: warm-up micro-steps over every shape, then timed
micro-steps with each phase synchronized (text, vae, fwd+bwd), plus one optimizer step. Model FLOPs are counted
analytically (padded tokens included, checkpoint recompute excluded; checked once against FlopCounterMode).

CUDA_VISIBLE_DEVICES=0 uv run python moss/tools/probe_sft_throughput.py [--batches 1 2 4 8 16 24 32]
    [--checkpoint none all 21] [--compile off on] [--sdpa cudnn default] [--resolution 512|1024]
    [--seq-pad 128] [--compile-dynamic static] [--out runs/moss-sft-probe]
    [--trace B CKPT COMPILE SDPA]   # one config under torch.profiler -> profiles/moss-sft-b{B}-...json
Note: setup memory counts AdamW (2 fp32 states); the trainer with Muon + EMA holds about as much, plus Muon
momentum once the first step has run (~17 GB above the probe's peak at 1024, B=2).
    [--text-parity]            # batched vs per-prompt encoder on 64 captions, then exit
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import glob
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from moss.train_sft import configure, sample_t

MODEL = "checkpoints/moss-image-base"
COLUMNS = ("tags", "10_word_summary", "brief_summary", "midjourney_style_summary", "long_caption", "scene_graph")
SHAPES = ((512, 512), (448, 576), (576, 448), (384, 640))
PEAK_TFLOPS = 2250.0  


def caption_pool(n=2048, seed=0):
    import pyarrow.parquet as pq
    from dataloaders.caption_stream import format_caption
    rng, pool = random.Random(seed), []
    files = sorted(glob.glob("trainer_samples_v3/source=*/samples.parquet"))
    for path in files:
        rows = pq.read_table(path, columns=list(COLUMNS)).slice(1000, 4 * n // len(files)).to_pylist()
        for row in rng.sample(rows, n // len(files)):
            column = rng.choice([c for c in COLUMNS if row[c]])
            text = row[column]
            if column == "tags":
                tags = text.split(",")
                rng.shuffle(tags)
                text = ",".join(tags).lstrip()
            pool.append(format_caption(text, "compact_json") if text.lstrip().startswith("{") else text)
    rng.shuffle(pool)
    return pool


def forward_flops(config, n_img, padded, batch):
    """Forward FLOPs of one batched pass (matmuls + attention)."""
    d, f, t = config.dim, config.ffn_dim, config.text_dim
    seq = n_img + padded
    block = 2 * seq * (4 * d * d + 3 * d * f) + 4 * seq * seq * d
    text_block = 2 * padded * (4 * d * d + 3 * d * f) + 4 * padded * padded * d
    text = 2 * padded * t * d + 2 * text_block
    return batch * (config.num_layers * block + text + 2 * n_img * config.latent_channels * d * 2)


class Probe:
    def __init__(self, device, extra, resolution=512):
        from moss.model.conditioners import Encoder, TextEncoder
        from moss.model.transformer import load_transformer
        self.device, self.extra, self.resolution = device, extra, resolution
        self.model = load_transformer(Path(MODEL) / "transformer", device, dtype=torch.float32)
        self.model.requires_grad_(True).train()
        self.params = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(self.params, lr=1e-12, betas=(0.9, 0.99), weight_decay=0.0, fused=True)
        for p in self.params:
            p.grad = torch.zeros_like(p)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.encoder = TextEncoder(MODEL, device, batched=True)
        self.vae = Encoder(MODEL, device)
        self.pool = caption_pool()
        self.cursor = 0
        self.timestep = {"mean": 0.0, "std": 1.0, "shift": 3.0}

    def captions(self, n):
        out = [self.pool[(self.cursor + i) % len(self.pool)] for i in range(n)]
        self.cursor += n
        return out

    def set_mode(self, checkpoint, compiled, sdpa):
        for block in self.model.layers:
            block.__dict__.pop("forward", None)
        configure(self.model, {"batched_forward": True, "grad_checkpoint": checkpoint, "compile": compiled,
                               "sdpa": sdpa, **self.extra})

    def micro_step(self, batch, shape, captions):
        sync = torch.cuda.synchronize
        marks = [time.perf_counter()]
        contexts = self.encoder(captions)
        sync(); marks.append(time.perf_counter())
        pixels = torch.rand(batch, 3, shape[1], shape[0], device=self.device) * 2 - 1
        x0 = self.vae(pixels)
        sync(); marks.append(time.perf_counter())
        t = sample_t(batch, self.timestep, self.device)
        eps = torch.randn_like(x0)
        tt = t.view(-1, 1, 1, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = self.model((1 - tt) * x0 + tt * eps, t * 1000, contexts).float()
        F.mse_loss(v, eps - x0).backward()
        sync(); marks.append(time.perf_counter())
        n_img = x0.shape[-1] * x0.shape[-2]
        padded = self.model.text_padding(n_img, max(c.shape[0] for c in contexts))
        flops = 3 * forward_flops(self.model.config, n_img, padded, batch)
        return {"text": marks[1] - marks[0], "vae": marks[2] - marks[1], "fwd_bwd": marks[3] - marks[2],
                "flops": flops, "text_tokens": sum(c.shape[0] for c in contexts), "padded": padded}

    def optimizer_step(self):
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return time.perf_counter() - start

    def run(self, batch, checkpoint, compiled, sdpa, warmup, steps):
        """``warmup`` passes over the same ``steps`` micro-batches that are then timed, so every (shape, text
        padding) the timed steps see is already compiled."""
        self.set_mode(checkpoint, compiled, sdpa)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        shapes = [(w * self.resolution // 512, h * self.resolution // 512) for w, h in SHAPES]
        specs = [(shapes[i % len(shapes)], self.captions(batch)) for i in range(steps)]
        started = time.perf_counter()
        for _ in range(warmup):
            for shape, captions in specs:
                self.micro_step(batch, shape, captions)
                self.optimizer.zero_grad(set_to_none=True)
        warm_seconds = time.perf_counter() - started
        records = []
        for i, (shape, captions) in enumerate(specs):
            records.append(self.micro_step(batch, shape, captions))
            if i < steps - 1:
                self.optimizer.zero_grad(set_to_none=True)
        optim = self.optimizer_step()
        total = {k: sum(r[k] for r in records) for k in ("text", "vae", "fwd_bwd", "flops", "text_tokens")}
        images = batch * steps
        step_seconds = total["text"] + total["vae"] + total["fwd_bwd"]
        return {"batch": batch, "checkpoint": checkpoint, "compile": compiled, "sdpa": sdpa, "images": images,
                "images_per_s": images / step_seconds, "ms_per_image": 1000 * step_seconds / images,
                "text_ms_per_image": 1000 * total["text"] / images, "vae_ms_per_image": 1000 * total["vae"] / images,
                "fwd_bwd_ms_per_image": 1000 * total["fwd_bwd"] / images,
                "fwd_bwd_tflops": total["flops"] / total["fwd_bwd"] / 1e12,
                "mfu_fwd_bwd": total["flops"] / total["fwd_bwd"] / 1e12 / PEAK_TFLOPS,
                "mfu_step": total["flops"] / step_seconds / 1e12 / PEAK_TFLOPS,
                "optimizer_s": optim, "warmup_s": warm_seconds,
                "mean_text_tokens": total["text_tokens"] / images,
                "peak_gib": torch.cuda.max_memory_allocated() / 2**30}


def check_flops(probe):
    from torch.utils.flop_counter import FlopCounterMode
    probe.set_mode("none", False, "default")
    contexts = probe.encoder(probe.captions(2))
    x = torch.randn(2, 128, 32, 32, device=probe.device)
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), counter:
        probe.model(x, torch.full((2,), 500.0, device=probe.device), contexts)
    padded = probe.model.text_padding(1024, max(c.shape[0] for c in contexts))
    analytic = forward_flops(probe.model.config, 1024, padded, 2)
    print(f"[probe] forward FLOPs: counted {counter.get_total_flops() / 1e12:.3f} T, "
          f"analytic {analytic / 1e12:.3f} T", flush=True)


def text_parity(probe):
    caps = probe.captions(64)
    worst, means = 1.0, []
    for i in range(0, 64, 16):
        a, b = probe.encoder(caps[i:i + 16], batched=False), probe.encoder(caps[i:i + 16], batched=True)
        for x, y in zip(a, b):
            cos = F.cosine_similarity(x.float(), y.float(), dim=-1)
            worst, means = min(worst, cos.min().item()), means + [cos.mean().item()]
    print(f"[probe] batched vs per-prompt text: min token cos {worst:.5f}, mean {sum(means) / len(means):.6f}")


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    mode = lambda r: (r["checkpoint"], r["compile"], r.get("sdpa", "default"))
    for key in sorted({mode(r) for r in results}, key=str):
        rows = sorted((r for r in results if mode(r) == key), key=lambda r: r["batch"])
        label = f"ckpt {key[0]}, compile {'on' if key[1] else 'off'}, sdpa {key[2]}"
        ax[0].plot([r["batch"] for r in rows], [r["images_per_s"] for r in rows], "o-", label=label)
        ax[1].plot([r["batch"] for r in rows], [100 * r["mfu_fwd_bwd"] for r in rows], "o-", label=label)
    ax[0].set(xscale="log", xlabel="per-GPU batch", ylabel="images/s (text + vae + fwd/bwd)", title="throughput")
    ax[1].set(xscale="log", xlabel="per-GPU batch", ylabel="% of 2.25 PFLOP/s", title="fwd+bwd MFU")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=110)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 24, 32])
    parser.add_argument("--checkpoint", nargs="+", default=["none", "all", "21"])
    parser.add_argument("--compile", nargs="+", default=["off", "on"], choices=["off", "on"])
    parser.add_argument("--sdpa", nargs="+", default=["cudnn"], choices=["default", "cudnn"])
    parser.add_argument("--warmup", type=int, default=2, help="passes over the timed micro-batches")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--out", default="runs/moss-sft-probe")
    parser.add_argument("--trace", nargs=4, metavar=("B", "CKPT", "COMPILE", "SDPA"))
    parser.add_argument("--resolution", type=int, default=512, help="bucket base (512 or 1024)")
    parser.add_argument("--seq-pad", type=int, default=128, help="0 = pad the text to a multiple of 64 instead")
    parser.add_argument("--compile-dynamic", default="static", choices=["static", "auto", "dynamic"])
    parser.add_argument("--text-parity", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    probe = Probe(device, {"seq_pad": args.seq_pad or None, "compile_dynamic": args.compile_dynamic}, args.resolution)
    print(f"[probe] setup peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
          f"(model + grads-capable + AdamW + encoder + VAE)", flush=True)
    if args.text_parity:
        text_parity(probe)
        return
    ckpt = lambda value: int(value) if value.isdigit() else value

    if args.trace:
        batch, checkpoint, compiled, sdpa = int(args.trace[0]), ckpt(args.trace[1]), args.trace[2] == "on", args.trace[3]
        probe.run(batch, checkpoint, compiled, sdpa, args.warmup, 3)
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            result = probe.run(batch, checkpoint, compiled, sdpa, 0, 3)
        Path("profiles").mkdir(exist_ok=True)
        path = Path("profiles") / f"moss-sft-b{batch}-ckpt{checkpoint}-compile{args.trace[2]}-{sdpa}.json"
        prof.export_chrome_trace(str(path))
        print(json.dumps(result, indent=2))
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
        print(f"[probe] trace {path}")
        return

    check_flops(probe)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done = {(r["batch"], str(r["checkpoint"]), r["compile"], r.get("sdpa", "default")) for r in results}
    modes = [(sdpa, compiled == "on", ckpt(checkpoint))
             for sdpa in args.sdpa for compiled in args.compile for checkpoint in args.checkpoint]
    for sdpa, compiled, checkpoint in modes:
        for batch in args.batches:
            if (batch, str(checkpoint), compiled, sdpa) in done:
                continue
            try:
                result = probe.run(batch, checkpoint, compiled, sdpa, args.warmup, args.steps)
            except torch.OutOfMemoryError:
                probe.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"[probe] B={batch} ckpt={checkpoint} compile={compiled} sdpa={sdpa}: OOM", flush=True)
                break
            results.append(result)
            results_path.write_text(json.dumps(results, indent=2))
            print(f"[probe] B={batch:>2} ckpt={str(checkpoint):>4} compile={'on ' if compiled else 'off'} "
                  f"sdpa={sdpa:<7} {result['images_per_s']:6.2f} img/s  {result['ms_per_image']:6.1f} ms/img "
                  f"(text {result['text_ms_per_image']:.1f} vae {result['vae_ms_per_image']:.1f} fwd/bwd "
                  f"{result['fwd_bwd_ms_per_image']:.1f})  fwd/bwd {result['fwd_bwd_tflops']:.0f} TFLOP/s "
                  f"MFU {100 * result['mfu_fwd_bwd']:.0f}%  optim {result['optimizer_s']:.2f}s  "
                  f"peak {result['peak_gib']:.0f} GiB", flush=True)
    plot(results, out / "throughput.png")
    best = max(r["images_per_s"] for r in results)
    pick = min((r for r in results if r["images_per_s"] >= 0.95 * best), key=lambda r: (r["batch"], r["ms_per_image"]))
    print(f"[probe] peak {best:.2f} img/s; smallest within 5%: B={pick['batch']} ckpt={pick['checkpoint']} "
          f"compile={pick['compile']} ({pick['images_per_s']:.2f} img/s, peak {pick['peak_gib']:.0f} GiB)")


if __name__ == "__main__":
    main()
