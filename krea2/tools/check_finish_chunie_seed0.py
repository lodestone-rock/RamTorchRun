"""CPU-only control-flow test: finisher cannot schedule training or seed1/2."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from krea2 import finish_chunie_seed0 as finish


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        jobs = []
        for method, gpu in [('lora', 1), ('sparse', 0)]:
            folder = root / method / 'seed0'
            folder.mkdir(parents=True)
            cfg = dict(seed=0, max_steps=300, peft_method=method,
                       parquet_dataloader={'caption_columns': {}, 'offset': 0})
            path = folder / 'config.json'
            path.write_text(json.dumps(cfg))
            jobs.append(dict(method=method, gpu=gpu, pid=123, create_time=0, config=str(path)))
            (root / method / 'selection.json').write_text(json.dumps(dict(selected_lr=.0001, tuning=[])))
        (root / 'seed0_only_handoff.json').write_text(json.dumps(dict(final_seeds=[0], jobs=jobs)))
        (root / 'calibration/ckpts').mkdir(parents=True)
        (root / 'calibration/ckpts/result.json').write_text('{}')
        ckpt = root / 'adapter.safetensors'
        ckpt.write_bytes(b'test')
        calls = []

        def evaluate(name, gpu, cfg):
            assert cfg['evaluation_only'] and not cfg['save_final']
            assert cfg['max_steps'] == cfg['steps_per_epoch'] == 32
            assert cfg['seed'] == 987654 and name.endswith('/seed0/test')
            assert cfg['parquet_dataloader']['parquet_sources']['chunie']['path'] == str(root/'data/test.parquet')
            calls.append(('eval', name, gpu))
            return {'evaluation_loss': .12}, ckpt

        def render(method, checkpoint, gpu, seed, **kwargs):
            assert seed == 0 and kwargs == {'resident': True}
            calls.append(('render', method, gpu))
            folder = root / f'renders/{method}_seed0'
            folder.mkdir(parents=True)
            Image.new('RGB', (1024, 512)).save(folder / 'grid.png')

        with (patch.object(finish, 'OUT', root),
              patch.object(finish, 'wait_for_trainer', return_value=({'global_step': 300}, ckpt)),
              patch.object(finish, 'run', side_effect=evaluate),
              patch.object(finish, 'render', side_effect=render),
              patch.object(finish, 'compare_hashes') as hashes):
            finish.main()
        assert len(calls) == 5 and ('render', 'base', 1) in calls
        assert hashes.call_count == 5
        assert json.loads((root/'comparison.json').read_text())['final_seeds'] == [0]
        assert json.loads((root/'seed0_finish_status.json').read_text())['phase'] == 'complete'
        with Image.open(root/'renders/base_lora_sparse_seed0.png') as image:
            assert image.size == (1024, 1536)
    print('PASS: evaluation-only configs, seed0-only scheduling, 3 resident renders, hashes, report/grid/status')


if __name__ == '__main__':
    main()