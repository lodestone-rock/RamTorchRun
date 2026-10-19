"""How slow is gradient / weight sync for the 5B MOSS transformer on this box (PCIe, no NVLink)?

Allocates every trainable MOSS parameter shape on each rank and times, per update:
  all_reduce   fp32 all-reduce of every grad (what DDP does once per update)
  reduce       ZeRO-1 grad sync: each grad reduced to its owner (greedy largest-first, as train_sft "zero")
  broadcast    ZeRO-1 weight sync: each owner broadcasts its updated weights
in fp32 and bf16, each issued three ways: ``loop`` (one blocking call per tensor), ``async`` (all calls
async_op=True, then wait) and ``coalesced`` (one dist._coalescing_manager group, a single NCCL group launch).

torchrun --nproc_per_node 4 moss/tools/probe_comms.py
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 moss/tools/probe_comms.py   # one same-socket pair
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node 4 moss/tools/probe_comms.py         # staged through host memory
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import time

import torch
import torch.distributed as dist

from moss.model.transformer import MOSSConfig, MOSSImageTransformer
from utils.zero_sharding import assign_owners


def shapes(config_path):
    raw = json.loads(open(config_path).read()) if config_path else {}
    config = MOSSConfig(**{k: raw[k] for k in MOSSConfig.__dataclass_fields__ if k in raw})
    with torch.device("meta"):
        model = MOSSImageTransformer(config)
    return [(n, tuple(p.shape)) for n, p in model.named_parameters()]


def timed(fn, repeats):
    fn()
    torch.cuda.synchronize()
    dist.barrier()
    best = float("inf")
    for _ in range(repeats):
        torch.cuda.synchronize()
        dist.barrier()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - started)
    return best


def issue(kind, tensors, owners, mode, device):
    def call(tensor, owner, async_op):
        if kind == "all_reduce":
            return dist.all_reduce(tensor, async_op=async_op)
        if kind == "reduce":
            return dist.reduce(tensor, dst=owner, async_op=async_op)
        return dist.broadcast(tensor, src=owner, async_op=async_op)

    if mode == "loop":
        for tensor, owner in zip(tensors, owners):
            call(tensor, owner, False)
    elif mode == "async":
        works = [call(tensor, owner, True) for tensor, owner in zip(tensors, owners)]
        for work in works:
            work.wait()
    else:
        with dist._coalescing_manager(device=device, async_ops=True) as manager:
            for tensor, owner in zip(tensors, owners):
                call(tensor, owner, False)
        manager.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="checkpoints/moss-image-sft/transformer/config.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", default="loop,async,coalesced")
    parser.add_argument("--dtypes", default="float32,bfloat16")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    named = shapes(args.config if os.path.exists(args.config) else None)
    owners = assign_owners([torch.Size(s).numel() for _, s in named], world)
    numel = sum(torch.Size(s).numel() for _, s in named)
    if rank == 0:
        print(f"[comms] world {world}, {len(named)} tensors, {numel / 1e9:.2f}B params; "
              f"P2P {'off' if os.environ.get('NCCL_P2P_DISABLE') == '1' else 'on'}; "
              f"devices {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}", flush=True)
    results = []
    for dtype_name in args.dtypes.split(","):
        dtype = getattr(torch, dtype_name)
        tensors = [torch.ones(s, device=device, dtype=dtype) for _, s in named]
        gigabytes = numel * tensors[0].element_size() / 1e9
        for kind in ("all_reduce", "reduce", "broadcast"):
            for mode in args.modes.split(","):
                seconds = timed(lambda: issue(kind, tensors, owners, mode, device), args.repeats)
                results.append({"dtype": dtype_name, "op": kind, "mode": mode, "seconds": seconds,
                                "payload_gb": gigabytes, "algo_gbps": gigabytes / seconds})
                if rank == 0:
                    print(f"  {dtype_name:9s} {kind:10s} {mode:9s} {seconds:7.3f}s  "
                          f"{gigabytes:5.1f} GB payload  {gigabytes / seconds:6.1f} GB/s algo", flush=True)
        del tensors
        torch.cuda.empty_cache()
    if rank == 0:
        best = {}
        for r in results:
            key = (r["dtype"], r["op"])
            best[key] = min(best.get(key, float("inf")), r["seconds"])
        for dtype_name in args.dtypes.split(","):
            zero = best[(dtype_name, "reduce")] + best[(dtype_name, "broadcast")]
            print(f"[comms] {dtype_name}: DDP all_reduce {best[(dtype_name, 'all_reduce')]:.2f}s vs "
                  f"ZeRO-1 reduce + broadcast {zero:.2f}s per update", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
