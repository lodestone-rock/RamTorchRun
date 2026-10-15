"""Chunie sparse/LoRA pilot on trainer_samples_v3 at 512px, four GPUs.

GPU0 inflight calibration then sparse lr A, GPU1 sparse lr B (after the mask),
GPU2/GPU3 LoRA lr A/B. Same trainer for calibration, fitting and held-out loss.
Completed runs are reused; existing jobs are never signalled.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from dataloaders.parquet_dataloader import _get_s3_client

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/k2-chunie-peft-v3-512'
SOURCE = '/mnt/datapool_u2/lodestone/mass_caption_v2/out/trainer_samples_v3/source=e621'
S3 = dict(endpoint_url='http://localhost:9000',
          credential_file=str(ROOT / 'credential/local-rustfs-readonly.json'),
          strip_prefix='/mnt/nas_buckets/')
LRS = [1e-4, 5e-5]
STEPS = 1000
CALIBRATION_STEPS = 10
VALIDATION_IMAGES = 4
VALIDATION_BATCHES = 32
CAPTIONS = {
    'tags': dict(weight=2.0, is_tag_based=True),
    'midjourney_style_summary': dict(weight=2.0, is_tag_based=False),
    '10_word_summary': dict(weight=1.0, is_tag_based=False),
    'long_caption': dict(weight=1.0, is_tag_based=False),
}


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2) + '\n'
    if path.exists() and path.read_text() != text:
        raise ValueError(f'Refusing changed experiment definition: {path}')
    path.write_text(text)


def image_bytes(path):
    bucket, _, key = path[len(S3['strip_prefix']):].partition('/')
    return _get_s3_client(S3).get_object(Bucket=bucket, Key=key)['Body'].read()


def prepare():
    table = ds.dataset(SOURCE, format='parquet').to_table(filter=ds.field('artist') == 'chunie')
    # Two v3 rows carry bare 'e621/xx/yy/<md5>' paths that neither disk nor S3 resolves.
    loadable = [i for i, p in enumerate(table.column('file_path').to_pylist())
                if p.startswith(S3['strip_prefix'])]
    unloadable = table.num_rows - len(loadable)
    table = table.take(loadable)
    rows = table.to_pylist()
    hashes = [hashlib.sha256(image_bytes(r['file_path'])).hexdigest() for r in rows]
    # Drop byte-identical duplicates BEFORE splitting, keeping the first row.
    first = {}
    for i, h in enumerate(hashes):
        first.setdefault(h, i)
    unique = sorted(first.values())
    order = sorted(unique, key=lambda i: hashlib.sha256(('20261014' + hashes[i]).encode()).hexdigest())
    splits = dict(train=order[VALIDATION_IMAGES:], validation=order[:VALIDATION_IMAGES])
    write(OUT/'split.json', dict(source=SOURCE, rows=len(rows), dropped_unloadable=unloadable,
          unique_images=len(unique),
          splits={k: [dict(path=rows[i]['file_path'], sha256=hashes[i]) for i in ids]
                  for k, ids in splits.items()}))
    for split, ids in splits.items():
        path = OUT/f'data/{split}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        subset = table.take(sorted(ids))
        if path.exists():
            assert pq.read_table(path).equals(subset)
        else:
            pq.write_table(subset, path)
    return {k: len(v) for k, v in splits.items()}


def config(method, seed, steps, lr, split='train', mask=None, checkpoint=None, optimizer='adamw'):
    cfg = json.loads((ROOT/'krea2/configs/train_mass_lora_v2_e621.json').read_text())
    cfg = {k: v for k, v in cfg.items() if not k.startswith('_comment')}
    cfg.update(seed=seed, initial_global_step=0, parallelism='resident', devices=['cuda:0'],
               chunks_per_stage=None, slots_per_step=1, per_slot_batch=1, n_microbatches=4,
               slot_allowlist=['chunie'], max_slots=1, min_samples_per_slot=1,
               steps_per_epoch=steps, max_steps=steps, min_slot_steps=0,
               peft_method=method, lora_rank=8, lora_alpha=8, bank_dtype='fp32',
               sparse_calibration='inflight', sparse_candidate_multiplier=2,
               bank_checkpoint=None, sparse_mask=str(mask) if mask else None,
               bank_state_device='cpu', weight_decay=0, lr=lr, warmup=20,
               grad_ckpt=True, eval_interval=0, save_every_n_steps=250, log_every_n_steps=1,
               save_final=True, paired_step_rng=True,
               mmdit_checkpoint=str(ROOT/'checkpoints/krea2/fullft_step_49600.safetensors'))
    if optimizer == 'lion':
        cfg.update(optimizer='lion', lion=dict(betas=[0.9, 0.99]))
    p = cfg['parquet_dataloader']
    p.update(parquet_sources={'chunie': dict(path=str(OUT/f'data/{split}.parquet'), n_samples=None)},
             caption_columns=CAPTIONS, base_resolution=[512], base_resolution_weights=[1.0],
             resolution_step=64, s3_image_source=S3, tag_drop_percentage=0.0,
             offset=0, num_workers=1, prefetch_factor=2)
    if split != 'train':
        cfg.update(evaluation_only=True, save_final=False, save_every_n_steps=0,
                   uncond_ratio=0, seed=987654, warmup=0)
        p['shuffle_tags'] = False
        cfg['sparse_checkpoint' if method == 'sparse' else 'bank_checkpoint'] = str(checkpoint)
    return cfg


def run(name, gpu, cfg):
    folder = OUT/name
    cfg.update(ckpt_path=str(folder/'ckpts'), preview_path=str(folder/'previews'))
    write(folder/'config.json', cfg)
    result = folder/'ckpts/result.json'
    if not result.exists():
        if (folder/'ckpts').exists():
            archive = folder/f'failed_{time.time_ns()}'
            archive.mkdir()
            for entry in ('ckpts', 'previews', 'train.log'):
                if (folder/entry).exists():
                    (folder/entry).rename(archive/entry)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   PYTHONUNBUFFERED='1', PYTHONHASHSEED='0')
        print(f'[{name}] start on GPU{gpu}', flush=True)
        with (folder/'train.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT/'krea2/train_mass_lora.py'), str(folder/'config.json')],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    data = json.loads(result.read_text())
    assert data['global_step'] == cfg['max_steps']
    print(f'[{name}] done: {json.dumps({k: data.get(k) for k in ("global_step", "evaluation_loss")})}', flush=True)
    return data, folder/f"ckpts/bank_step_{cfg['max_steps']}_final.safetensors"


def compare_hashes(a, b):
    a, b = OUT/a/'ckpts/input_hashes.jsonl', OUT/b/'ckpts/input_hashes.jsonl'
    if a.read_bytes() != b.read_bytes():
        raise AssertionError(f'Unmatched training inputs: {a} vs {b}')


def render(label, method, checkpoint, gpu):
    from PIL import Image
    from safetensors.torch import load_file, save_file
    from krea2.tools.export_lora_bank import slice_slot
    prompts = [
        'A fully clothed anthropomorphic fox explorer wearing a green jacket, standing beside a mountain lake at sunrise.',
        'A cheerful anthropomorphic rabbit in a yellow raincoat holding an umbrella on a rainy city street.',
        'An anthropomorphic wolf musician in a formal suit playing a violin in a warmly lit concert hall.',
        'An anthropomorphic cat librarian in a sweater reading a book by a window, autumn leaves outside.',
        'A red bicycle leaning against a white garden wall, flowering vines, afternoon sunlight.',
        'A cozy wooden cabin beside a snowy forest, warm light in the windows, twilight.',
        'A bowl of oranges on a blue tablecloth beside a ceramic teapot, still life.',
        'A sailing boat in a quiet harbor with colorful waterfront buildings at dawn.',
    ]
    manifest = [dict(prompt='by chunie, ' + p, seed=4000+i) for i, p in enumerate(prompts)]
    write(OUT/'render_prompts.json', manifest)
    folder = OUT/f'renders/{label}'
    folder.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT/'krea2/inference.py'), '--config',
           str(ROOT/'krea2/configs/train_pipeline_lora.json'), '--mmdit-checkpoint',
           str(ROOT/'checkpoints/krea2/fullft_step_49600.safetensors'),
           '--device', 'cuda:0', '--batch-size', '1',
           '--steps', '28', '--guidance', '4.5', '--mu', '1.15',
           '--width', '512', '--height', '512', '--prompts-file',
           str(OUT/'render_prompts.json'), '--out-dir', str(folder)]
    if method == 'sparse':
        cmd += ['--no-lora', '--sparse-checkpoint', str(checkpoint)]
    elif method == 'lora':
        exported = folder/'adapter.safetensors'
        save_file(slice_slot(load_file(checkpoint), 0), exported)
        cmd += ['--lora-checkpoint', str(exported), '--lora-rank', '8', '--lora-alpha', '8']
    else:
        cmd += ['--no-lora']
    if not (folder/'manifest.json').exists():
        with (folder/'render.log').open('w') as log:
            subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = json.loads((folder/'manifest.json').read_text())
    assert len(rows) == len(manifest)
    thumbs = []
    for row, expected in zip(rows, manifest):
        assert row['prompt'] == expected['prompt'] and row['seed'] == expected['seed']
        with Image.open(folder/row['image']) as image:
            thumbs.append(image.convert('RGB').resize((256, 256)))
    grid = Image.new('RGB', (4*256, 2*256))
    for i, image in enumerate(thumbs):
        grid.paste(image, ((i % 4)*256, (i//4)*256))
    grid.save(folder/'grid.png')
    return folder/'grid.png'


def evaluate_checkpoint(family, lr, step, gpu):
    """Held-out loss + render grid for one saved bank_step_N checkpoint.

    ``family`` is the run folder (lora, lora_lion, sparse); the adapter type is
    sparse for sparse/ and LoRA for everything else.
    """
    method = 'sparse' if family == 'sparse' else 'lora'
    ckpt = OUT/f'{family}/lr_{lr}/ckpts/bank_step_{step}.safetensors'
    name = f'{family}/lr_{lr}/step{step}_validation'
    if method == 'sparse':
        mask = OUT/f'calibration/ckpts/bank_step_{CALIBRATION_STEPS}_final.safetensors'
        cfg = config(method, 1234, VALIDATION_BATCHES, lr, 'validation', mask, ckpt)
        # Score through an empty LoRA bank: fits beside a running trainer.
        cfg.update(peft_method='lora', sparse_mask=None, bank_checkpoint=None)
    else:
        cfg = config(method, 1234, VALIDATION_BATCHES, lr, 'validation', None, ckpt)
    val, _ = run(name, gpu, cfg)
    grid = render(f'{family}_lr_{lr}_step{step}', method, ckpt, gpu)
    return dict(method=family, lr=lr, step=step, validation_loss=val['evaluation_loss'], grid=str(grid))


def evaluate_all_step(step, family=None):
    """Every arm (of ``family``, if given) with a bank_step_N checkpoint,
    round-robin over the 4 GPUs, sweep arms first."""
    arms = sorted(((p.parts[-4], float(p.parts[-3][3:]))
                   for p in OUT.glob(f'{family or "*"}/lr_*/ckpts/bank_step_{step}.safetensors')),
                  key=lambda a: (a[1] in LRS, a[0], a[1]))
    queues = [arms[gpu::4] for gpu in range(4)]

    def per_gpu(gpu):
        return [evaluate_checkpoint(method, lr, step, gpu) for method, lr in queues[gpu]]

    with ThreadPoolExecutor(max_workers=4) as pool:
        reports = [r for j in [pool.submit(per_gpu, g) for g in range(4)] for r in j.result()]
    reports.sort(key=lambda r: (r['method'], r['lr']))

    from PIL import Image
    panels = [OUT/'renders/base/grid.png'] + [Path(r['grid']) for r in reports]
    combined = Image.new('RGB', (1024, 512*len(panels)))
    for i, path in enumerate(panels):
        with Image.open(path) as image:
            combined.paste(image, (0, i*512))
    tag = family or 'all'
    combined.save(OUT/f'renders/{tag}_step{step}.png')
    write(OUT/f'step{step}_{tag}_eval.json', dict(
        reports=reports, panel_order=['base'] + [f"{r['method']}_lr_{r['lr']}" for r in reports]))
    for r in reports:
        print(f"step{step} {r['method']:6s} lr={r['lr']:.3g} validation_loss={r['validation_loss']:.5f}", flush=True)


def evaluate_step(step):
    """Held-out loss + renders for an intermediate checkpoint while training continues.

    Runs on the LoRA GPUs (2: lr A, 3: lr B), which have the most free memory.
    """
    def per_gpu(lr, gpu):
        reports = [evaluate_checkpoint(method, lr, step, gpu) for method in ('lora', 'sparse')]
        compare_hashes(f'sparse/lr_{lr}/step{step}_validation', f'lora/lr_{lr}/step{step}_validation')
        return reports

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(per_gpu, LRS[0], 2), pool.submit(per_gpu, LRS[1], 3)]
        reports = [r for j in jobs for r in j.result()]
    base = render('base', 'base', None, 2)

    from PIL import Image
    panels = [base] + [Path(r['grid']) for r in reports]
    combined = Image.new('RGB', (1024, 512*len(panels)))
    for i, path in enumerate(panels):
        with Image.open(path) as image:
            combined.paste(image, (0, i*512))
    combined.save(OUT/f'renders/base_lora_sparse_step{step}.png')
    write(OUT/f'step{step}_eval.json', dict(
        reports=reports, panel_order=['base'] + [f"{r['method']}_lr_{r['lr']}" for r in reports]))
    for r in reports:
        print(f"step{step} {r['method']:6s} lr={r['lr']:.0e} validation_loss={r['validation_loss']:.5f}", flush=True)


def lora_sweep(lrs, steps=(250, 500, 750), optimizer='adamw'):
    """LoRA-only LR sweep, one GPU per LR; held-out loss + renders at each checkpoint.

    AdamW arms live under lora/, Lion arms under lora_lion/; both see the
    same per-step inputs as the lora/lr_0.0001 baseline.
    """
    family = 'lora' if optimizer == 'adamw' else f'lora_{optimizer}'

    def arm(lr, gpu):
        name = f'{family}/lr_{lr}'
        _, final = run(name, gpu, config('lora', 1234, STEPS, lr, optimizer=optimizer))
        rows = []
        for step in (*steps, STEPS):
            ckpt = final if step == STEPS else OUT/f'{name}/ckpts/bank_step_{step}.safetensors'
            suffix = 'validation' if step == STEPS else f'step{step}_validation'
            val, _ = run(f'{name}/{suffix}', gpu,
                         config('lora', 1234, VALIDATION_BATCHES, lr, 'validation', None, ckpt))
            label = f'{family}_lr_{lr}' + ('' if step == STEPS else f'_step{step}')
            rows.append(dict(lr=lr, step=step, validation_loss=val['evaluation_loss'],
                             grid=str(render(label, 'lora', ckpt, gpu))))
        return rows

    with ThreadPoolExecutor(max_workers=len(lrs)) as pool:
        jobs = [pool.submit(arm, lr, gpu) for gpu, lr in enumerate(lrs)]
        reports = [r for j in jobs for r in j.result()]
    for lr in lrs:
        compare_hashes('lora/lr_0.0001', f'{family}/lr_{lr}')

    from PIL import Image
    for step in (*steps, STEPS):
        panels = [OUT/'renders/base/grid.png'] + [Path(r['grid']) for r in reports if r['step'] == step]
        combined = Image.new('RGB', (1024, 512*len(panels)))
        for i, path in enumerate(panels):
            with Image.open(path) as image:
                combined.paste(image, (0, i*512))
        combined.save(OUT/f'renders/{family}_sweep_step{step}.png')
    tag = '_'.join(f'{lr:g}' for lr in lrs)
    write(OUT/f'{family}_sweep_{tag}.json', dict(reports=reports, panel_order=['base'] + [str(lr) for lr in lrs]))
    for r in reports:
        print(f"{family} lr={r['lr']:.3g} step={r['step']} validation_loss={r['validation_loss']:.5f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--eval-step', type=int, help='Evaluate saved bank_step_N checkpoints and exit')
    ap.add_argument('--lora-sweep', type=float, nargs='+', help='LoRA-only LRs, one GPU each (max 4)')
    ap.add_argument('--eval-all-step', type=int, help='Evaluate every arm\'s bank_step_N on GPUs 0-3 and exit')
    ap.add_argument('--optimizer', choices=('adamw', 'lion'), default='adamw', help='For --lora-sweep')
    ap.add_argument('--family', help='Restrict --eval-all-step to one run folder (lora, lora_lion, sparse)')
    args = ap.parse_args()
    if args.eval_all_step:
        tag = args.family or 'all'
        with (OUT/f'eval_{tag}_step{args.eval_all_step}.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            evaluate_all_step(args.eval_all_step, args.family)
        return
    if args.lora_sweep:
        with (OUT/f'lora_sweep_{args.optimizer}.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lora_sweep(args.lora_sweep, optimizer=args.optimizer)
        return
    if args.eval_step:
        with (OUT/f'eval_step{args.eval_step}.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            evaluate_step(args.eval_step)
        return
    with (OUT/'launcher.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        full_run(args)


def full_run(args):
    counts = prepare()
    write(OUT/'protocol.json', dict(
        data=SOURCE, split=counts, resolution=512, captions=CAPTIONS,
        saliency='GPU top-tail per backward; signed sparse microbatch sum; absolute step top-tail; '
                 f'average {CALIBRATION_STEPS} sparse steps; global top-k',
        calibration_steps=CALIBRATION_STEPS, rank=8, alpha=8, master_dtype='fp32', weight_decay=0,
        lr_grid=LRS, steps=STEPS, warmup=20, effective_batch=4, validation_batches=VALIDATION_BATCHES,
        base='checkpoints/krea2/fullft_step_49600.safetensors',
        gpu_assignment={'calibration': 0, f'sparse_{LRS[0]}': 0, f'sparse_{LRS[1]}': 1,
                        f'lora_{LRS[0]}': 2, f'lora_{LRS[1]}': 3}))
    print(f'Prepared chunie v3 split {counts}', flush=True)
    if args.prepare_only:
        return

    def arm(method, lr, gpu, mask_future=None):
        mask = mask_future.result()[1] if mask_future else None
        name = f'{method}/lr_{lr}'
        metrics, ckpt = run(name, gpu, config(method, 1234, STEPS, lr, mask=mask))
        val, _ = run(name+'/validation', gpu,
                     config(method, 1234, VALIDATION_BATCHES, lr, 'validation', mask, ckpt))
        grid = render(f'{method}_lr_{lr}', method, ckpt, gpu)
        return dict(method=method, lr=lr, validation_loss=val['evaluation_loss'],
                    training=metrics, checkpoint=str(ckpt), grid=str(grid))

    with ThreadPoolExecutor(max_workers=5) as pool:
        cal = pool.submit(run, 'calibration', 0, config('sparse', 1234, CALIBRATION_STEPS, LRS[0]))
        jobs = [pool.submit(arm, 'lora', LRS[0], 2), pool.submit(arm, 'lora', LRS[1], 3),
                pool.submit(arm, 'sparse', LRS[0], 0, cal), pool.submit(arm, 'sparse', LRS[1], 1, cal)]
        reports = [j.result() for j in jobs]
    for lr in LRS:
        compare_hashes(f'sparse/lr_{lr}', f'lora/lr_{lr}')
        compare_hashes(f'sparse/lr_{lr}/validation', f'lora/lr_{lr}/validation')
    base = render('base', 'base', None, 0)

    from PIL import Image
    panels = [base] + [Path(r['grid']) for r in reports]
    combined = Image.new('RGB', (1024, 512*len(panels)))
    for i, path in enumerate(panels):
        with Image.open(path) as image:
            combined.paste(image, (0, i*512))
    combined.save(OUT/'renders/base_lora_sparse.png')
    write(OUT/'comparison.json', dict(
        reports=reports, input_hashes_match=True, panel_order=['base'] + [f"{r['method']}_lr_{r['lr']}" for r in reports],
        calibration=json.loads((OUT/'calibration/ckpts/result.json').read_text())))
    for r in reports:
        print(f"{r['method']:6s} lr={r['lr']:.0e} validation_loss={r['validation_loss']:.5f}", flush=True)
    print('COMPARISON COMPLETE', flush=True)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    main()
