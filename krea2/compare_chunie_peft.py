"""Reproducible one-artist sparse/LoRA pilot; GPU0 sparse, GPU1 LoRA.

Runs the SAME trainer for calibration, fitting and fixed-noise evaluation.
Existing jobs are never signalled by this launcher. Completed runs are reused.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import fcntl
import time

import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/k2-chunie-peft-inflight'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2) + '\n'
    if path.exists() and path.read_text() != text:
        raise ValueError(f'Refusing changed experiment definition: {path}')
    path.write_text(text)


def prepare():
    source = '/mnt/datapool_u2/lodestone/mass_caption_v2/out/trainer_samples/source=e621'
    table = ds.dataset(source, format='parquet').to_table(filter=ds.field('artist') == 'chunie')
    rows = table.to_pylist()
    # Hash actual bytes: reject duplicate images BEFORE splitting.
    hashes = [hashlib.sha256(Path(r['file_path']).read_bytes()).hexdigest() for r in rows]
    if len(rows) != 8 or len(set(hashes)) != 8:
        raise ValueError('Expected eight distinct chunie images')
    order = sorted(range(8), key=lambda i: hashlib.sha256(('20260907' + hashes[i]).encode()).hexdigest())
    splits = dict(train=order[:6], validation=order[6:7], test=order[7:])
    write(OUT/'split.json', dict(source=source, splits={k:[dict(path=rows[i]['file_path'],sha256=hashes[i]) for i in ids] for k,ids in splits.items()}))
    for split, ids in splits.items():
        path = OUT/f'data/{split}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        subset = table.take(ids)
        if path.exists():
            assert pq.read_table(path).equals(subset)
        else:
            pq.write_table(subset, path)


def config(method, seed, steps, lr, split='train', mask=None, checkpoint=None):
    cfg = json.loads((ROOT/'krea2/configs/train_mass_lora_v2_e621.json').read_text())
    cfg = {k:v for k,v in cfg.items() if not k.startswith('_comment')}
    cfg.update(seed=seed, initial_global_step=0, parallelism='resident', devices=['cuda:0'],
               chunks_per_stage=None, slots_per_step=1, per_slot_batch=1, n_microbatches=4,
               slot_allowlist=['chunie'], max_slots=1, min_samples_per_slot=1,
               steps_per_epoch=steps, max_steps=steps, min_slot_steps=0,
               peft_method=method, lora_rank=8, lora_alpha=8, bank_dtype='fp32',
               sparse_calibration='inflight', sparse_candidate_multiplier=2,
               bank_checkpoint=None, sparse_mask=str(mask) if mask else None,
               bank_state_device='cpu', weight_decay=0, lr=lr, warmup=20,
               grad_ckpt=True, eval_interval=0, save_every_n_steps=0, log_every_n_steps=1,
               save_final=True, paired_step_rng=True,
               mmdit_checkpoint=str(ROOT/'checkpoints/krea2/fullft_step_49600.safetensors'))
    p=cfg['parquet_dataloader']
    p.update(parquet_sources={'chunie':dict(path=str(OUT/f'data/{split}.parquet'),n_samples=None)},
             offset=0, num_workers=1, prefetch_factor=2)
    if split != 'train':
        cfg.update(evaluation_only=True, save_final=False, uncond_ratio=0,
                   seed=987654, warmup=0)
        p['shuffle_tags']=False
        p['caption_columns']={'brief_summary':dict(weight=1,is_tag_based=False)}
        if method == 'sparse':
            cfg['sparse_checkpoint']=str(checkpoint)
        else:
            cfg['bank_checkpoint']=str(checkpoint)
    return cfg


def run(name, gpu, cfg):
    folder=OUT/name
    cfg.update(ckpt_path=str(folder/'ckpts'),preview_path=str(folder/'previews'))
    write(folder/'config.json',cfg)
    result=folder/'ckpts/result.json'
    if not result.exists():
        if (folder/'ckpts').exists():
            archive = folder/f'failed_{time.time_ns()}'
            archive.mkdir()
            for entry in ('ckpts','previews','train.log'):
                if (folder/entry).exists():
                    (folder/entry).rename(archive/entry)
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                 PYTHONUNBUFFERED='1', PYTHONHASHSEED='0')
        with (folder/'train.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'krea2/train_mass_lora.py'),str(folder/'config.json')],
                           cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    data=json.loads(result.read_text())
    assert data['global_step']==cfg['max_steps']
    checkpoint=folder/f"ckpts/bank_step_{cfg['max_steps']}_final.safetensors"
    return data, checkpoint


def compare_hashes(a,b):
    a,b=OUT/a/'ckpts/input_hashes.jsonl',OUT/b/'ckpts/input_hashes.jsonl'
    if a.read_bytes()!=b.read_bytes():
        raise AssertionError(f'Unmatched training inputs: {a} vs {b}')


def render(method, checkpoint, gpu, seed, *, resident=False):
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
    manifest = [dict(prompt='by chunie, ' + p, seed=4000+i) for i,p in enumerate(prompts)]
    write(OUT/'render_prompts.json', manifest)
    folder = OUT/f'renders/{method}_seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT/'krea2/inference.py'), '--config',
           str(ROOT/'krea2/configs/train_pipeline_lora.json'), '--mmdit-checkpoint',
           str(ROOT/'checkpoints/krea2/fullft_step_49600.safetensors'),
           '--offload', '--device', 'cuda:0', '--batch-size', '1',
           '--steps', '28', '--guidance', '4.5', '--mu', '1.15',
           '--width', '1024', '--height', '1024', '--prompts-file',
           str(OUT/'render_prompts.json'), '--out-dir', str(folder)]
    if resident:
        cmd.remove('--offload')
    if method == 'sparse':
        cmd += ['--no-lora','--sparse-checkpoint',str(checkpoint)]
    elif method == 'lora':
        exported=folder/'adapter.safetensors'
        save_file(slice_slot(load_file(checkpoint),0),exported)
        cmd += ['--lora-checkpoint',str(exported),'--lora-rank','8','--lora-alpha','8']
    else:
        cmd += ['--no-lora']
    if not (folder/'manifest.json').exists():
        with (folder/'render.log').open('w') as log:
            subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),
                           OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT,check=True)
    rows=json.loads((folder/'manifest.json').read_text())
    assert len(rows)==len(manifest)
    thumbs=[]
    for row, expected in zip(rows,manifest):
        assert row['prompt']==expected['prompt'] and row['seed']==expected['seed']
        with Image.open(folder/row['image']) as image:
            assert image.size==(1024,1024)
            thumbs.append(image.convert('RGB').resize((256,256)))
    grid=Image.new('RGB',(4*256,2*256))
    for i,image in enumerate(thumbs):
        grid.paste(image,((i%4)*256,(i//4)*256))
    grid.save(folder/'grid.png')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--smoke-then-run',action='store_true',help='Run the smoke gate then the full comparison')
    args=ap.parse_args()
    if args.smoke and args.smoke_then_run:
        ap.error('Choose --smoke or --smoke-then-run')
    if (OUT/'seed0_only_handoff.json').exists():
        raise RuntimeError('Run shortened to seed0; use krea2/finish_chunie_seed0.py, not the multi-seed launcher')
    prepare()
    write(OUT/'protocol.json',dict(train_images=6,validation_images=1,test_images=1,
          saliency='GPU top-tail per backward; signed sparse microbatch sum; absolute step top-tail; average ten sparse steps; global top-k',
          sparse_candidate_multiplier=2,approximate_saliency=True,
          calibration_steps=10,calibration_updates_base=False,rank=8,alpha=8,
          trainable_scalars=29326336,master_dtype='fp32',weight_decay=0,
          lr_grid=[2e-5,1e-4,2e-4],tuning_steps=100,final_steps=300,seeds=[0,1,2],
          effective_batch=4,validation_batches=16,test_batches=32,
          gpu_assignment=dict(sparse=0,lora=1),timing_contended=True))
    print('Prepared fixed 6/1/1 image split; sparse GPU0, LoRA GPU1; attention sweep untouched.', flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        if args.smoke or args.smoke_then_run:
            cal=pool.submit(run,'smoke/calibration',0,config('sparse',1234,2,1e-4))
            lo=pool.submit(run,'smoke/lora',1,config('lora',1234,2,1e-4))
            _,mask=cal.result(); _,lora=lo.result()
            _,sparse=run('smoke/sparse',0,config('sparse',1234,2,1e-4,mask=mask))
            compare_hashes('smoke/sparse','smoke/lora')
            for method,ckpt,gpu in [('sparse',sparse,0),('lora',lora,1)]:
                run('smoke/eval_'+method,gpu,config(method,1234,2,1e-4,'validation',mask,ckpt))
            compare_hashes('smoke/eval_sparse','smoke/eval_lora')
            write(OUT/'smoke/passed.json',dict(passed=True,paired_training_inputs=True,paired_eval_inputs=True))
            print('SMOKE PASS: actual K2 calibration, paired updates and held-out evaluation',flush=True)
            if not args.smoke_then_run:
                return
        # Shared, training-only calibration mask: fixed across all candidates/seeds.
        _,mask=run('calibration',0,config('sparse',1234,10,1e-4))
        def experiment(method,gpu):
            tuning=[]
            for lr in [2e-5,1e-4,2e-4]:
                name=f'{method}/tune_{lr}'
                _,ckpt=run(name,gpu,config(method,1234,100,lr,mask=mask))
                metrics,_=run(name+'/validation',gpu,config(method,1234,16,lr,'validation',mask,ckpt))
                tuning.append(dict(lr=lr,loss=metrics['evaluation_loss']))
            lr=min(tuning,key=lambda r:r['loss'])['lr']
            write(OUT/method/'selection.json',dict(tuning=tuning,selected_lr=lr))
            finals=[]
            for seed in [0,1,2]:
                name=f'{method}/seed{seed}'
                metrics,ckpt=run(name,gpu,config(method,seed,300,lr,mask=mask))
                test,_=run(name+'/test',gpu,config(method,seed,32,lr,'test',mask,ckpt))
                finals.append(dict(seed=seed,test_loss=test['evaluation_loss'],training=metrics,checkpoint=str(ckpt)))
            return dict(method=method,lr=lr,tuning=tuning,finals=finals)
        futures=[pool.submit(experiment,m,g) for m,g in [('sparse',0),('lora',1)]]
        reports=[f.result() for f in futures]
    for lr in [2e-5,1e-4,2e-4]:
        compare_hashes(f'sparse/tune_{lr}',f'lora/tune_{lr}')
    for seed in [0,1,2]:
        compare_hashes(f'sparse/seed{seed}',f'lora/seed{seed}')
        compare_hashes(f'sparse/seed{seed}/test',f'lora/seed{seed}/test')
    for report in reports:
        for final in report['finals']:
            final['adapter_bytes']=Path(final['checkpoint']).stat().st_size
    write(OUT/'comparison.json',dict(reports=reports,input_hashes_match=True,
          calibration=json.loads((OUT/'calibration/ckpts/result.json').read_text()),
          caveat='One held-out image; artist-specific pilot, not a general superiority claim. Timing contended.'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[]
        for report,gpu in zip(reports,[0,1]):
            def render_method(report=report,gpu=gpu):
                for final in report['finals']:
                    render(report['method'],final['checkpoint'],gpu,final['seed'])
            jobs.append(pool.submit(render_method))
        for job in jobs:
            job.result()
    render('base',None,0,0)
    from PIL import Image
    panels = [OUT/'renders/base_seed0/grid.png'] + [
        OUT/f'renders/{method}_seed0/grid.png' for method in ('lora','sparse')]
    combined=Image.new('RGB',(1024,512*3))
    for i,path in enumerate(panels):
        with Image.open(path) as image:
            combined.paste(image,(0,i*512))
    combined.save(OUT/'renders/base_lora_sparse_seed0.png')
    print('COMPARISON COMPLETE',flush=True)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'launcher.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()