"""concept_marker.py — the ``<concept:NAME>`` prompt marker.

One tiny parser shared by the concept-LoRA trainer (cache build + training
loop), the launcher's renderer and ``inference.py``, so training and prompting
speak the same syntax:

    "<concept:chunie> male, anthro"  ->  ("male, anthro", ["chunie"])

The marker is a discrete switch that activates the trained concept-embedding
token (see ``krea2/model/tag_embed.py`` / ``train_concept_lora.py``); the
stripped remainder goes to the text encoder as usual. Rules:

- The marker itself is ALWAYS removed from the text, known or not — a prompt
  never leaks ``<concept:...>`` into the encoder. Unknown names are stripped
  too and reported via the ``unknown`` return value instead of an error, so a
  prompt rendered without the matching checkpoint degrades gracefully.
- No marker at all is the common case: ``parse`` returns the text unchanged
  (modulo the collapse regexes not firing) and an empty list, which maps to a
  masked-out concept row — the same bit-exact no-op the tag-embed design
  rests on.
- Prose artist mentions ("Drawn by chunie.") are NOT markers and pass through
  untouched. The marker is explicit, user-written syntax only.

No torch dependency: this is importable from CPU tools and data prep.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Name charset is deliberately generous but excludes '>' and ':' (so a
# hypothetical nested "<concept:a:b" cannot smuggle a second marker body):
# letters, digits, underscore, hyphen and spaces. The trainer lowercases
# artist names on both sides, so lookup is case-insensitive.
_MARKER_RE = re.compile(r"<concept:\s*([A-Za-z0-9_\- ]+?)\s*>")

# Leftover separator debris after a strip: a leading comma (possibly with the
# space it absorbed), a doubled comma, or a comma glued to the new neighbor.
_DEBRIS_RES = (
    re.compile(r"^\s*,\s*"),          # marker was the first segment
    re.compile(r",\s*,+"),            # marker sat between two commas
    re.compile(r",\s*$"),             # marker was the last segment
)


def parse_concept_markers(text: str, known=None):
    """Parse *text*, strip markers, return (clean_text, names, unknown).

    *known* is an optional iterable of valid concept names (matched
    case-insensitively). Names found in the text but absent from *known* are
    still stripped and still returned — they are listed in *unknown* so the
    caller can warn; when *known* is None every parsed name is "known".

    Returns ``(clean_text, names, unknown)`` where *names* is the
    deduplicated, order-preserving list of marker names (lowercased) and
    *unknown* the subset not in *known*.
    """
    if not text or "<concept:" not in text:
        return text, [], []

    known_set = {str(k).strip().lower() for k in known} if known is not None else None

    names: list[str] = []
    unknown: list[str] = []

    def _sub(m: re.Match) -> str:
        raw = m.group(1).strip()
        name = raw.lower()
        if name and name not in names:
            names.append(name)
        if known is not None and name not in known_set:
            if name not in unknown:
                unknown.append(name)
        return ""  # the marker never stays in the text

    clean = _MARKER_RE.sub(_sub, text)

    for pattern in _DEBRIS_RES:
        clean = pattern.sub("" if pattern is _DEBRIS_RES[0] else ",", clean)

    # Collapse whitespace runs the strip may have created and trim the ends.
    clean = re.sub(r"[ \t]{2,}", " ", clean).strip()

    return clean, names, unknown


def marker_row(names, name_to_id: dict[str, int], max_len: int = 1):
    """Map parsed marker names to a (tag_ids, tag_mask) concept row.

    ``name_to_id`` is the trainer's vocabulary (lowercased name -> row id).
    A marker whose name is missing from the map yields a masked slot rather
    than a KeyError — callers that care check ``mask.any()``.

    Returns ``(ids, mask)`` shaped ``[1, max_len]`` (int64 / bool).
    """
    import torch

    ids = torch.zeros(1, max_len, dtype=torch.long)
    mask = torch.zeros(1, max_len, dtype=torch.bool)
    for i, name in enumerate(names[:max_len]):
        row = name_to_id.get(str(name).strip().lower())
        if row is not None:
            ids[0, i] = row
            mask[0, i] = True
    return ids, mask
