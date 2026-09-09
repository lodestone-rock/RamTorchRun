"""CPU checks for orthogonal sweep construction and strict row reuse."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from collections import Counter
import itertools
import json
from pathlib import Path
import tempfile

from PIL import Image

from krea2.sweep_attention import BASE, TURBO, design, reuse_rows, verify


def main():
    old_rows, old_policies = design(2)
    rows, policies = design(4)
    assert len(rows) == len(set(map(tuple, rows))) == 64
    assert all(Counter(r[c] for r in rows) == dict.fromkeys(range(4), 16) for c in range(7))
    for a, b in itertools.combinations(range(7), 2):
        assert Counter((r[a], r[b]) for r in rows) == dict.fromkeys(
            itertools.product(range(4), repeat=2), 4)
    assert all(p in policies for p in old_policies)
    assert [rows.index(r) for r in old_rows] == [0, 1, 4, 5, 16, 17, 20, 21]
    assert all(len(p) == 28 and all(len(h) == 12 and all(0 <= v < 48 for v in h)
                                   for h in p) for p in policies)
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        src, dst = root / "source", root / "dest"
        src.mkdir(); dst.mkdir()
        prompts = [dict(prompt=f'landscape {i}', seed=100+i) for i in range(20)]
        shared = dict(checkpoint_files={}, prompts_sha256="test", steps=8, guidance=0,
                      mu=1.15, window=11, full_heads=12, calibrated=False,
                      layer_bands=[[4*i, 4*i+3] for i in range(7)])
        for folder, row_list in ((src, old_rows[:1]), (dst, rows)):
            (folder / "design.json").write_text(json.dumps(dict(shared, rows=row_list)))
            (folder / "prompts.json").write_text(json.dumps(prompts))
        (src / "policy_00.json").write_text(json.dumps(policies[0]))
        case = src / "row_00"
        case.mkdir()
        Image.new("RGB", (1024, 1024)).save(case / "image.png")
        entries = [dict(prompt=p["prompt"], seed=p["seed"], source_metadata=p,
                        mmdit_checkpoint=str(BASE), lora_checkpoint=str(TURBO),
                        lora_scale=1.0, steps=8, guidance=0, mu=1.15, image="image.png",
                        attention=dict(mode="local", full_query_heads=12, full_head_ids=None,
                                       layer_head_ids=policies[0], window=11)) for p in prompts]
        (case / "manifest.json").write_text(json.dumps(entries))
        assert reuse_rows(dst, src, prompts, policies) == 1
        assert reuse_rows(dst, src, prompts, policies) == 1
        assert (dst / "row_00/reuse.json").exists()
        assert not (case / "reuse.json").exists()
        verify(dst / "row_00", prompts, policies[0])
        entries[0]["seed"] = -1
        (case / "manifest.json").write_text(json.dumps(entries))
        try:
            reuse_rows(dst, src, prompts, policies)
        except AssertionError:
            pass
        else:
            raise AssertionError("Corrupt source seed accepted")
        definition = dict(shared, rows=rows, window=13)
        (dst / "design.json").write_text(json.dumps(definition))
        try:
            reuse_rows(dst, src, prompts, policies)
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched recipe accepted")
    print("PASS L8 subset, L64 orthogonality/budgets, exact reuse, idempotence, mismatch rejection")


if __name__ == "__main__":
    main()