"""Split a bank into interleaved sub-banks without changing adapter weights."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def split_bank(source: Path, output: Path, shards: int = 2):
    source, output = Path(source).resolve(), Path(output).resolve()
    metadata = Path(str(source) + '.json')
    if not metadata.exists():
        metadata = source.parent / 'slots.json'
    meta = json.loads(metadata.read_text())
    if meta.get('bank_checkpoint') != source.name:
        raise ValueError('Metadata does not describe the requested checkpoint')
    n = meta['n_slots']
    if shards < 1 or shards > n:
        raise ValueError('shards must be between 1 and n_slots')
    for field in ('slot_names', 'slot_steps', 'slot_samples'):
        if len(meta[field]) != n:
            raise ValueError(f'Invalid {field} length')
    if len(set(meta['slot_names'])) != n:
        raise ValueError('Duplicate artist names')
    if any(not isinstance(s, int) or s < 0 for s in meta['slot_steps']):
        raise ValueError('Invalid step counters')
    paths = [output / f'shard{i}' / 'bank_init.safetensors' for i in range(shards)]
    if any(p.parent.exists() for p in paths):
        raise FileExistsError('Refusing to overwrite an existing shard directory')
    with safe_open(str(source), framework='pt', device='cpu') as src:
        keys = list(src.keys())
        if not keys:
            raise ValueError('Empty bank')
        for k in keys:
            shape = src.get_slice(k).get_shape()
            if not k.endswith(('.lora_A_bank', '.lora_B_bank')):
                raise ValueError(f'Unexpected non-bank tensor: {k}')
            rank_axis = 1 if k.endswith('.lora_A_bank') else 2
            if len(shape) != 3 or shape[0] != n or shape[rank_axis] != meta['rank']:
                raise ValueError(f'Invalid bank geometry: {k}: {shape}')
        for i, path in enumerate(paths):
            indices = list(range(i, n, shards))
            idx = torch.tensor(indices)
            sd = {k: src.get_tensor(k).index_select(0, idx).contiguous() for k in keys}
            if not all(torch.isfinite(t).all().item() for t in sd.values()):
                raise ValueError('Non-finite bank weights')
            path.parent.mkdir(parents=True)
            save_file(sd, str(path) + '.tmp')
            os.replace(str(path) + '.tmp', path)
            del sd
            # Read-back every tensor, not just headers or a sample of slots.
            with safe_open(str(path), framework='pt', device='cpu') as dst:
                if set(dst.keys()) != set(keys):
                    raise AssertionError('Output tensor key mismatch')
                for k in keys:
                    if not torch.equal(dst.get_tensor(k), src.get_tensor(k).index_select(0, idx)):
                        raise AssertionError(f'Shard {i} differs from source: {k}')
            m = dict(meta)
            for field in ('slot_names', 'slot_steps', 'slot_samples'):
                m[field] = [meta[field][s] for s in indices]
            m.update(n_slots=len(indices), bank_checkpoint=path.name,
                     source_checkpoint=str(source), source_slot_indices=indices,
                     optimizer_moments_restored=False)
            for p in (Path(str(path) + '.json'), path.parent / 'slots.json'):
                p.write_text(json.dumps(m, indent=2) + '\n')
            print(f'Verified shard {i}: {len(indices)} artists, '
                  f'{min(m["slot_steps"])}..{max(m["slot_steps"])} updates -> {path}', flush=True)
    return paths


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--shards', type=int, default=2)
    args = ap.parse_args()
    split_bank(args.source, args.output, args.shards)