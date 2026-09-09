"""Continue the old pipeline banks as four independent resident GPU banks.

First split the source checkpoints with tools/split_lora_bank.py. This
launcher validates their metadata, uses the new dataset, and preserves the
per-artist counters. Global steps count NEW worker updates, not old pipeline
steps, which are not comparable after sharding. Adam moments start fresh.
"""
from __future__ import annotations

import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krea2.run_mass_lora_single_gpu import ROOT, make_config, run_child, write_json


def main():
    control = ROOT / 'runs/k2-mass-lora-resume-pipeline-v2'
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / 'launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (control / 'launched.json').exists():
        raise FileExistsError('Continuation was already launched')
    jobs = []
    for gpu in range(4):
        source = 'danbooru' if gpu < 2 else 'e621'
        bank = control / f'initial-banks/{source}/shard{gpu % 2}/bank_init.safetensors'
        meta = json.loads(Path(str(bank) + '.json').read_text())
        if not bank.is_file() or meta['n_slots'] != 128:
            raise ValueError(f'Incomplete shard: {bank}')
        directory = ROOT / f'runs/k2-mass-lora-v2-{source}/continued-single-gpu-trainer-samples-v2/worker{gpu}-bank0'
        config = directory / 'config.json'
        if config.exists():
            raise FileExistsError(config)
        cfg = make_config(source, meta['slot_names'], directory, seats=4)
        original = json.loads((ROOT / f'krea2/configs/train_mass_lora_v2_{source}.json').read_text())
        if meta['slot_names'] != original['slot_allowlist'][gpu % 2::2]:
            raise ValueError('Shard artists do not match the original experiment')
        if (meta['rank'], meta['alpha']) != (cfg['lora_rank'], cfg['lora_alpha']):
            raise ValueError('Shard rank/alpha does not match config')
        cfg.update(bank_checkpoint=str(bank), initial_global_step=0,
                   initialization_lock=str(control / 'initialization.lock'),
                   strict_bank_resume=True,
                   continuation_source=meta['source_checkpoint'],
                   continuation_initial_slot_steps=meta['slot_steps'])
        write_json(config, cfg)
        jobs.append((gpu, config))
        print(f'GPU{gpu}: {source}, {min(meta["slot_steps"])}..{max(meta["slot_steps"])} '
              'historical updates -> target300 total', flush=True)
    write_json(control / 'launched.json', {'workers': {str(g): [str(c)] for g, c in jobs},
                                         'optimizer_moments_restored': False})

    def worker(job):
        gpu, config = job
        result = run_child(config, gpu)
        if result['returncode'] != 0 or result['oom']:
            raise RuntimeError(f'GPU{gpu}: {result}')
        report = json.loads((config.parent / 'ckpts/result.json').read_text())
        if not report['target_met']:
            raise RuntimeError(f'GPU{gpu} did not reach target')
        return gpu

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for future in concurrent.futures.as_completed([pool.submit(worker, j) for j in jobs]):
            try:
                print('Completed GPU', future.result(), flush=True)
            except Exception as exc:
                failures.append(str(exc))
                print('FAILED:', exc, flush=True)
    write_json(control / 'finished.json', {'success': not failures, 'failures': failures})
    if failures:
        raise RuntimeError('One or more continuation workers failed')


if __name__ == '__main__':
    main()