"""CPU check: shard weights, names and counters stay aligned; reject bad metadata."""
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from safetensors.torch import save_file, load_file
from krea2.tools.split_lora_bank import split_bank


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / 'bank.safetensors'
        sd = {'layer.lora_A_bank': torch.randn(5, 2, 7),
              'layer.lora_B_bank': torch.randn(5, 9, 2)}
        save_file(sd, str(source))
        meta = dict(n_slots=5, rank=2, alpha=2., bank_checkpoint=source.name,
                    slot_names=list('abcde'), slot_steps=[101, 202, 303, 404, 505],
                    slot_samples=[7, 8, 9, 10, 11])
        (root / 'slots.json').write_text(json.dumps(meta))
        paths = split_bank(source, root / 'out')
        for i, path in enumerate(paths):
            actual = load_file(str(path))
            m = json.loads(Path(str(path) + '.json').read_text())
            for k in sd:
                assert torch.equal(actual[k], sd[k][i::2])
            for field in ('slot_names', 'slot_steps', 'slot_samples'):
                assert m[field] == meta[field][i::2]
        try:
            split_bank(source, root / 'out')
        except FileExistsError:
            pass
        else:
            raise AssertionError('Overwrite accepted')
        meta['bank_checkpoint'] = 'wrong.safetensors'
        (root / 'slots.json').write_text(json.dumps(meta))
        try:
            split_bank(source, root / 'bad')
        except ValueError:
            pass
        else:
            raise AssertionError('Mismatched checkpoint metadata accepted')
    print('All split-bank checks PASS')


if __name__ == '__main__':
    main()