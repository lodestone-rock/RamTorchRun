"""Wait for existing seed0 trainers, evaluate/render, exit. Never starts training."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import json
from pathlib import Path
import time

import psutil
from PIL import Image
from safetensors import safe_open

from krea2.compare_chunie_peft import OUT, compare_hashes, render, run, write


def status(name, **fields):
    path = OUT / f'{name}_status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(updated=time.time(), **fields), indent=2) + '\n')
    temp.replace(path)
    print(name, fields, flush=True)


def wait_for_trainer(job):
    """Wait for the exact PID to exit, not merely for its result file to appear."""
    try:
        proc = psutil.Process(job['pid'])
        if proc.create_time() == job['create_time']:
            if job['config'] not in proc.cmdline():
                raise RuntimeError('Recorded trainer has an unexpected command')
            while proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                if proc.create_time() != job['create_time']:
                    break
                time.sleep(5)
    except psutil.NoSuchProcess:
        pass
    folder = Path(job['config']).parent
    result = json.loads((folder / 'ckpts/result.json').read_text())
    if result['global_step'] != 300 or result['slot_steps_min'] != 300:
        raise RuntimeError(f'{job["method"]} did not finish 300 seed0 updates')
    ckpt = folder / 'ckpts/bank_step_300_final.safetensors'
    meta = json.loads(Path(str(ckpt) + '.json').read_text())
    if (meta['seed'] != 0 or meta['slot_steps'] != [300]
            or meta['peft_method'] != job['method']):
        raise RuntimeError('Not the expected completed seed0 checkpoint')
    with safe_open(ckpt, framework='pt') as f:
        if not f.keys():
            raise RuntimeError('Empty checkpoint')
    return result, ckpt


def evaluation_config(job, checkpoint):
    # Freeze the actual training recipe, not a template another run may edit.
    cfg = copy.deepcopy(json.loads(Path(job['config']).read_text()))
    if cfg['seed'] != 0 or cfg['max_steps'] != 300:
        raise ValueError('Handoff must reference the original seed0 config')
    cfg.update(evaluation_only=True, save_final=False, uncond_ratio=0,
               seed=987654, warmup=0, max_steps=32, steps_per_epoch=32,
               initial_global_step=0, min_slot_steps=0, eval_interval=0,
               save_every_n_steps=0, bank_checkpoint=None)
    cfg.pop('peft_training_state', None)
    p = cfg['parquet_dataloader']
    p.update(parquet_sources={'chunie': dict(path=str(OUT/'data/test.parquet'), n_samples=None)},
             offset=0, shuffle_tags=False,
             caption_columns={'brief_summary': dict(weight=1, is_tag_based=False)})
    cfg['sparse_checkpoint' if job['method'] == 'sparse' else 'bank_checkpoint'] = str(checkpoint)
    return cfg


def finish_arm(job):
    method, gpu = job['method'], job['gpu']
    label = 'seed0_' + method
    try:
        status(label, phase='waiting_for_training', pid=job['pid'], gpu=gpu)
        metrics, checkpoint = wait_for_trainer(job)
        selection = json.loads((OUT / method / 'selection.json').read_text())
        status(label, phase='held_out_evaluation', gpu=gpu)
        test, _ = run(f'{method}/seed0/test', gpu, evaluation_config(job, checkpoint))
        status(label, phase='rendering_adapter', gpu=gpu)
        render(method, checkpoint, gpu, 0, resident=True)
        if method == 'lora':
            # LoRA is ahead; use GPU1 for the base while sparse finishes on0.
            status(label, phase='rendering_base_control', gpu=gpu)
            render('base', None, gpu, 0, resident=True)
        report = dict(method=method, lr=selection['selected_lr'], tuning=selection['tuning'],
                      finals=[dict(seed=0, test_loss=test['evaluation_loss'], training=metrics,
                                   checkpoint=str(checkpoint), adapter_bytes=checkpoint.stat().st_size)])
        write(OUT / method / 'seed0_only_result.json', report)
        status(label, phase='complete_gpu_released', gpu=gpu)
        return report
    except Exception as exc:
        status(label, phase='failed', gpu=gpu, error=str(exc))
        raise


def main():
    manifest = json.loads((OUT / 'seed0_only_handoff.json').read_text())
    jobs = manifest['jobs']
    if (manifest['final_seeds'] != [0] or len(jobs) != 2
            or {(j['method'], j['gpu']) for j in jobs} != {('lora', 1), ('sparse', 0)}):
        raise ValueError('Invalid seed0-only handoff')
    status('seed0_finish', phase='running', final_seeds=[0])
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            reports = list(pool.map(finish_arm, jobs))
        for name in ['tune_2e-05', 'tune_0.0001', 'tune_0.0002', 'seed0', 'seed0/test']:
            compare_hashes('sparse/' + name, 'lora/' + name)
        write(OUT / 'comparison.json', dict(
            reports=reports, input_hashes_match=True, final_seeds=[0],
            calibration=json.loads((OUT / 'calibration/ckpts/result.json').read_text()),
            caveat='User shortened to one seed. One held-out image; exploratory only. '
                   'All final renders use resident inference.'))
        combined = Image.new('RGB', (1024, 512 * 3))
        for i, method in enumerate(['base', 'lora', 'sparse']):
            with Image.open(OUT / f'renders/{method}_seed0/grid.png') as image:
                if image.size != (1024, 512):
                    raise ValueError('Unexpected preview grid size')
                combined.paste(image, (0, i * 512))
        combined.save(OUT / 'renders/base_lora_sparse_seed0.png')
        status('seed0_finish', phase='complete', final_seeds=[0], gpus_released=[0, 1])
    except Exception as exc:
        status('seed0_finish', phase='failed', error=str(exc))
        raise


if __name__ == '__main__':
    with (OUT / 'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()