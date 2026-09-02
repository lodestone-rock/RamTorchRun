"""tag_vocab.py — booru tag vocabulary and prompt -> tag-id matching.

A closed tag vocabulary is small enough to index DIRECTLY: `engram_viz`
measured 315,966 distinct tags over 1.16M booru tag lists, and order-1 is
permutation-invariant, so there is no ordering decision to make and no reason
to hash. `TagMatcher` maps a prompt to a list of vocabulary ids; the model side
(`krea2/model/tag_embed.py`) turns those ids into DiT tokens.

Matching runs in TWO passes, because the same table has to serve two very
different prompt styles:

  1. comma segments — split on commas/newlines and look the whole segment up.
     This is SDXL-style `1girl, solo, long hair` prompting, and it is also
     exactly what the training `tags` column looks like. High precision.
  2. word-trie longest match — for segments that are NOT a whole tag (i.e.
     natural language), scan for the longest tag phrase starting at each word.
     Word-level rather than character-level so `cat` never fires inside
     `category`, and non-overlapping so `long hair` beats `hair`.

Both passes normalize through `normalize_tag`, so `Long_Hair`, `LONG HAIR` and
` long   hair ` are one id. That matters more than it sounds: a user typing a
tag with the wrong case is the common failure mode, and a missed tag is a
silently weaker conditioning signal rather than a visible error.

Building the vocabulary (re-runnable; ALWAYS write a new version rather than
overwriting, since renumbering ids invalidates every trained table):

    uv run python utils/tag_vocab.py --out checkpoints/tag_vocab/tags_v1.parquet \
        --corpus /path/to/source=danbooru /path/to/source=e621 \
        --train-corpus /path/to/tag_samples_clean

`--corpus` decides the ID SPACE (use the biggest booru corpus available so ids
are stable), `--train-corpus` is reported alongside it so you can see how much
of the table will actually receive gradients before committing to a min_count.
"""
from __future__ import annotations

import argparse
import os
import unicodedata

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Characters that end a phrase for the free-text scan. Tags themselves contain
# '(', ')', ':', '-', '.' and "'" (e.g. "modeus (helltaker)", "re:shimashima"),
# so those are NOT separators — only structural punctuation is.
_SEGMENT_SPLIT = str.maketrans({c: "\n" for c in ",;\n\r\t|"})

# Trailing/leading noise stripped from a word before a trie lookup. Kept
# deliberately small: over-stripping merges distinct tags.
_EDGE_PUNCT = '.!?"\u201c\u201d\u2018\u2019'


def normalize_tag(s: str) -> str:
    """Canonical surface form of a tag.

    NFKC-folds width/compatibility variants, casefolds (so `Long Hair` and
    `long hair` agree), maps `_` to space (danbooru exports use underscores,
    this corpus uses spaces), drops the backslash escapes boorus put in front
    of parentheses, and collapses runs of whitespace.
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\\", "").replace("_", " ")
    return " ".join(s.casefold().split())


def split_segments(text: str) -> list[str]:
    """Split a prompt into comma/newline separated segments."""
    return [s for s in (p.strip() for p in text.translate(_SEGMENT_SPLIT).split("\n")) if s]


class TagVocab:
    """An ordered tag list plus its normalized-form lookup.

    ``forms[i]`` is the canonical (already normalized) text of id ``i``.
    ``counts[i]`` is its corpus frequency, kept only for reporting and for
    frequency-ordered previews — nothing in training depends on it.
    """

    def __init__(self, forms: list[str], counts=None, name: str = ""):
        self.forms = forms
        self.counts = (
            np.zeros(len(forms), dtype=np.int64) if counts is None
            else np.asarray(counts, dtype=np.int64)
        )
        self.name = name
        self.ids = {f: i for i, f in enumerate(forms)}
        if len(self.ids) != len(forms):
            raise ValueError("tag vocabulary contains duplicate normalized forms")

    def __len__(self) -> int:
        return len(self.forms)

    # -- persistence -------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "TagVocab":
        t = pq.read_table(path, columns=["tag", "count"])
        return cls(
            t.column("tag").to_pylist(),
            t.column("count").to_numpy(),
            name=os.path.basename(path),
        )

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if os.path.exists(path):
            raise FileExistsError(
                f"{path} already exists. Tag vocabularies are versioned, not "
                f"overwritten: renumbering ids silently invalidates every "
                f"TagEmbedder trained against the old file. Bump the version."
            )
        pq.write_table(
            pa.table({"tag": self.forms, "count": self.counts}),
            path, compression="zstd",
        )


# A single common English word can BE a rare tag: the corpus contains tags
# literally named "a" (6 uses), "best" (2) and "quality" (13). Firing those
# while scanning prose injects noise on every natural-language prompt, so the
# free-text pass only considers single-word tags that are common enough for
# their appearance in a sentence to plausibly be deliberate. Multi-word tags
# are exempt — "wooden table" is unambiguous however rare it is. This gates
# only the scan; an explicit comma segment still matches anything in the
# vocabulary, so rare artist and character tags are never blocked.
SCAN_MIN_COUNT = 100


class TagMatcher:
    """Prompt -> tag ids, via comma segments then a word-level longest match.

    The trie is a nested dict keyed by whole words; a node carries ``None`` ->
    id for the tag that ends there. Building it over 316k tags takes ~1s and
    ~250 MB, so it is constructed once per process and shared by the workers.
    """

    _END = None  # sentinel key holding the id of a tag ending at this node

    def __init__(self, vocab: TagVocab, build_trie: bool = True,
                 scan_min_count: int = SCAN_MIN_COUNT):
        self.vocab = vocab
        self.trie: dict | None = None
        if build_trie:
            self.trie = {}
            n = 0
            for i, form in enumerate(vocab.forms):
                words = form.split(" ")
                if len(words) == 1 and vocab.counts[i] < scan_min_count:
                    continue
                node = self.trie
                for w in words:
                    node = node.setdefault(w, {})
                node[self._END] = i
                n += 1
            self.scan_size = n

    # -- the two passes ----------------------------------------------------

    def _scan_words(self, words: list[str], out: list[int], seen: set[int]):
        """Longest-match, non-overlapping scan of a word list."""
        n, i = len(words), 0
        while i < n:
            node, best, j = self.trie, -1, i
            k = i
            while k < n:
                node = node.get(words[k])
                if node is None:
                    break
                k += 1
                tid = node.get(self._END)
                if tid is not None:
                    best, j = tid, k        # remember the LONGEST match so far
            if best >= 0:
                if best not in seen:
                    seen.add(best)
                    out.append(best)
                i = j                        # consume the whole matched phrase
            else:
                i += 1

    def match(self, text: str, free_text: bool = True) -> list[int]:
        """Tag ids present in *text*, in order of first appearance, deduped.

        ``free_text=False`` runs only the comma-segment pass, which is what the
        training `tags` column needs — it is already a clean comma list, and
        skipping the scan avoids inventing tags that the annotator did not
        assign (`solo` inside a `solo focus` segment, say).
        """
        out: list[int] = []
        seen: set[int] = set()
        for seg in split_segments(text):
            tid = self.vocab.ids.get(normalize_tag(seg))
            if tid is not None:
                if tid not in seen:
                    seen.add(tid)
                    out.append(tid)
                continue
            if not free_text or self.trie is None:
                continue
            words = [w.strip(_EDGE_PUNCT) for w in normalize_tag(seg).split(" ")]
            self._scan_words([w for w in words if w], out, seen)
        return out

    def encode(self, text: str, max_tags: int, free_text: bool = True):
        """-> (ids [max_tags] int64, mask [max_tags] bool), padded and truncated.

        Truncation keeps the FIRST ``max_tags`` matches. Tag order carries no
        information to the model (all tag tokens share one RoPE position), but
        it does decide who survives an overflow, so the caller should shuffle
        upstream if it wants the drop to be unbiased.
        """
        ids = np.zeros(max_tags, dtype=np.int64)
        mask = np.zeros(max_tags, dtype=bool)
        hits = self.match(text, free_text=free_text)[:max_tags]
        if hits:
            ids[: len(hits)] = hits
            mask[: len(hits)] = True
        return ids, mask


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _parquet_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    out = []
    for root, _d, files in os.walk(path):
        out.extend(os.path.join(root, f) for f in sorted(files)
                   if f.endswith(".parquet"))
    if not out:
        raise FileNotFoundError(f"no parquet files under {path}")
    return out


def count_tags(paths: list[str], column: str = "tags") -> tuple[dict[str, int], int]:
    """-> ({normalized tag: occurrences}, number of rows with any tag)."""
    counts: dict[str, int] = {}
    rows = 0
    for path in paths:
        for f in _parquet_files(path):
            if column not in pq.read_schema(f).names:
                print(f"  [skip] {f} has no '{column}' column")
                continue
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=8192, columns=[column]):
                for doc in batch.column(column).to_pylist():
                    if not doc:
                        continue
                    rows += 1
                    for raw in doc.split(","):
                        t = normalize_tag(raw)
                        if t:
                            counts[t] = counts.get(t, 0) + 1
    return counts, rows


def _histogram(counts: dict[str, int], label: str, cuts=(1, 2, 3, 5, 10, 25, 100)):
    if not counts:
        print(f"  {label}: empty")
        return
    arr = np.fromiter(counts.values(), dtype=np.int64, count=len(counts))
    total = arr.sum()
    print(f"  {label}: {len(arr):,} distinct, {total:,} occurrences")
    print(f"    {'min_count':>10} {'kept':>10} {'% of tags':>10} {'% traffic':>10}")
    for c in cuts:
        keep = arr >= c
        print(f"    {c:>10} {keep.sum():>10,} {100*keep.mean():>9.1f}% "
              f"{100*arr[keep].sum()/total:>9.2f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", nargs="+", required=True,
                    help="parquet dirs/files defining the ID SPACE")
    ap.add_argument("--train-corpus", nargs="*", default=[],
                    help="parquet dirs actually used for training (reported only)")
    ap.add_argument("--column", default="tags")
    ap.add_argument("--min-count", type=int, default=1,
                    help="drop tags rarer than this from the vocabulary")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"counting '{args.column}' over the vocabulary corpus...")
    counts, rows = count_tags(args.corpus, args.column)
    print(f"  {rows:,} rows with tags")
    _histogram(counts, "vocabulary corpus")

    if args.train_corpus:
        print("counting over the TRAINING corpus (how much of the table trains)...")
        tcounts, trows = count_tags(args.train_corpus, args.column)
        print(f"  {trows:,} rows with tags")
        _histogram(tcounts, "training corpus")
        overlap = sum(1 for t in counts if t in tcounts)
        print(f"  {overlap:,} of {len(counts):,} vocabulary tags "
              f"({100*overlap/max(len(counts),1):.1f}%) appear in training; "
              f"the rest never receive a gradient")

    kept = [t for t, c in counts.items() if c >= args.min_count]
    # Frequency order, ties by form, so id 0 is the most common tag and the
    # ordering is reproducible across runs of the same corpus.
    kept.sort(key=lambda t: (-counts[t], t))
    vocab = TagVocab(kept, [counts[t] for t in kept])
    vocab.save(args.out)
    print(f"\nwrote {args.out}: {len(vocab):,} tags (min_count={args.min_count})")
    print("  top 10:", ", ".join(vocab.forms[:10]))


if __name__ == "__main__":
    main()
