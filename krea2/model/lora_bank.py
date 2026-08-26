"""lora_bank.py — Many LoRAs at once off ONE frozen base (mass-LoRA training).

`lora.py` gives every `nn.Linear` a single low-rank delta (plus optional named
roles for TDM). This module gives it a **bank** of L independent deltas:

    lora_A_bank : [L, rank, in]
    lora_B_bank : [L, out, rank]

and evaluates them with a grouped `bmm` over a batch that is packed so each
slot's samples are CONTIGUOUS on dim 0:

    y  = x @ W^T + bias                      # one base GEMM, whole packed batch
    xg = x.reshape(S, -1, in)                # S contiguous groups
    y += ((xg @ A^T) @ B^T) * scale          # per-group delta

Slot i's samples only ever touch ``A[i]``/``B[i]``, so ``dL/dA[i]`` sees only
slot i's data — gradient isolation is structural, not enforced. Cost is the
same as a single LoRA over the same batch (one base GEMM, not S of them), and
under ``parallelism: "offload"`` the base weights cross PCIe once per step
instead of once per adapter, which is the whole reason not to hotswap.

Which slots are active is per-STEP module state, exactly like
`lora.set_lora_role` and `chunks.DiTHeadChunk.set_seq`:

    inject_lora_bank(dit, n_slots=64, rank=16)
    set_active_slots(dit, [3, 17, 40])       # then pack the batch as [.., 3, b]

It must not vary per MICROBATCH: `staggered_1b1f` keeps several microbatches in
flight, so per-microbatch state would race. Per-step state cannot — RamTorch's
engines swap tensors via ``functional_call`` and never touch attributes.

A slot that is absent from ``set_active_slots`` is never gathered, so its
gradient stays exactly zero. That is how a slot with no samples in the step's
resolution bucket sits the step out (see `utils/bank_optimizer.BankAdamW`,
which must then also skip its optimizer state).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRABankLinear(nn.Module):
    """`nn.Linear` drop-in carrying L independent LoRA adapters.

    ``active_slots`` selects which adapters the next forward applies, and
    implicitly how the batch is grouped: ``None`` means all L slots (so dim 0
    must be a multiple of L), a sequence of slot ids means those slots in that
    order (dim 0 must be a multiple of ``len(ids)``).

    Set it through :func:`set_active_slots`, never per microbatch.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        base_weight: torch.Tensor,
        base_bias: torch.Tensor | None,
        n_slots: int,
        rank: int,
        alpha: float,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_slots = n_slots
        self.rank = rank
        self.scale = alpha / rank

        # Frozen base — a buffer, so it rides along with .to()/.cuda() and with
        # RamTorch's chunk relocation but never reaches an optimizer.
        self.register_buffer("base_weight", base_weight.detach().clone())
        if base_bias is not None:
            self.register_buffer("base_bias", base_bias.detach().clone())
        else:
            self.base_bias = None

        # kaiming_uniform_(a=sqrt(5)) on a [rank, in] slice reduces exactly to
        # U(-1/sqrt(in), 1/sqrt(in)); doing it in closed form keeps the init to
        # one op per bank instead of L (and matches lora.py slice-for-slice,
        # which nn.init on the 3-D tensor would NOT — it would read fan_in as
        # rank * in).
        bound = 1.0 / math.sqrt(in_features)
        self.lora_A_bank = nn.Parameter(
            torch.empty(n_slots, rank, in_features).uniform_(-bound, bound)
        )
        # Zeros, so every slot starts as an exact no-op (delta W = 0).
        self.lora_B_bank = nn.Parameter(torch.zeros(n_slots, out_features, rank))

        self.active_slots: tuple[int, ...] | None = None
        # Per-device index tensors. Deliberately NOT a buffer: under weight
        # streaming the gathered banks live on the GPU while the masters are in
        # pinned CPU memory, so the index has to follow the *swapped* tensor's
        # device, and a buffer would be relocated/streamed like model state.
        self._idx_cache: dict[torch.device, torch.Tensor] = {}

    # -- slot selection ---------------------------------------------------

    def set_active_slots(self, slots):
        if slots is None:
            self.active_slots = None
        else:
            slots = tuple(int(s) for s in slots)
            if not slots:
                raise ValueError("active_slots must be non-empty (or None)")
            if any(not 0 <= s < self.n_slots for s in slots):
                raise ValueError(
                    f"slot ids {slots} out of range for n_slots={self.n_slots}"
                )
            if len(set(slots)) != len(slots):
                raise ValueError(f"duplicate slot ids in {slots}")
            self.active_slots = slots
        self._idx_cache.clear()

    @property
    def n_active(self) -> int:
        return self.n_slots if self.active_slots is None else len(self.active_slots)

    def _adapters(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        A, B = self.lora_A_bank, self.lora_B_bank
        if self.active_slots is None:
            return A, B
        idx = self._idx_cache.get(device)
        if idx is None:
            idx = torch.tensor(self.active_slots, dtype=torch.long, device=device)
            self._idx_cache[device] = idx
        return A.index_select(0, idx), B.index_select(0, idx)

    # -- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.base_weight, self.base_bias)
        A, B = self._adapters(x.device)
        s = A.shape[0]
        rows = x.shape[0]
        if rows % s != 0:
            raise RuntimeError(
                f"LoRABankLinear: leading dim {rows} is not a multiple of the "
                f"{s} active slot(s). The batch must be packed with each "
                f"slot's samples contiguous on dim 0."
            )
        # Group-contiguous reshape. Works for any leading layout as long as
        # samples stay ordered — including TextFusionTransformer's
        # (b l) n d, where dim 0 is b*l and sample i owns rows [i*l, (i+1)*l).
        xg = x.reshape(s, -1, self.in_features)
        h = torch.bmm(xg, A.transpose(1, 2).to(xg.dtype))
        d = torch.bmm(h, B.transpose(1, 2).to(h.dtype))
        return y + d.reshape(y.shape) * self.scale

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"slots={self.n_slots}, rank={self.rank}, scale={self.scale:.3f}"
        )


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def inject_lora_bank(
    model: nn.Module,
    n_slots: int,
    rank: int = 16,
    alpha: float | None = None,
    exclude_prefixes: tuple[str, ...] = (),
    exclude_patterns: tuple[str, ...] = (),
    verbose: bool = True,
) -> dict[str, LoRABankLinear]:
    """Replace every `nn.Linear` in *model* with a `LoRABankLinear`.

    Args:
        n_slots:          L, the number of independent adapters per layer.
        rank / alpha:     as in `lora.inject_lora` (alpha defaults to rank).
        exclude_prefixes: dotted module-name prefixes to leave untouched.
        exclude_patterns: substrings; any module whose dotted name contains one
                          is left untouched. The memory lever for mass
                          training — ``(".mlp.",)`` roughly halves the bank by
                          adapting attention only.

    Layers where the adapter would be no smaller than the weight it adapts
    (``rank >= min(in, out)``, e.g. K2's ``txtfusion.projector``, a
    ``Linear(12, 1)``) are skipped automatically and reported: L copies of a
    pointless adapter is L times the waste.
    """
    if alpha is None:
        alpha = float(rank)

    replaced: dict[str, LoRABankLinear] = {}
    degenerate: list[str] = []

    def _replace(parent: nn.Module, prefix: str):
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if any(full_name.startswith(ep) for ep in exclude_prefixes):
                continue
            if any(pat in full_name for pat in exclude_patterns):
                continue

            if isinstance(child, nn.Linear):
                if rank >= min(child.in_features, child.out_features):
                    degenerate.append(
                        f"{full_name} ({child.in_features}->{child.out_features})"
                    )
                    continue
                bank = LoRABankLinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    base_weight=child.weight,
                    base_bias=child.bias,
                    n_slots=n_slots,
                    rank=rank,
                    alpha=alpha,
                )
                setattr(parent, name, bank)
                replaced[full_name] = bank
            else:
                _replace(child, full_name)

    _replace(model, "")

    if verbose:
        print(
            f"[LoRA-bank] Replaced {len(replaced)} nn.Linear layers with "
            f"LoRABankLinear (slots={n_slots}, rank={rank}, alpha={alpha})."
        )
        if degenerate:
            print(
                f"[LoRA-bank] Skipped {len(degenerate)} layer(s) where "
                f"rank >= min(in, out): {', '.join(degenerate)}"
            )
    return replaced


def set_active_slots(model: nn.Module, slots):
    """Select which bank slots every `LoRABankLinear` in *model* applies.

    Call ONCE PER STEP, before ``pipe.step`` / ``pipe.infer``, and pack the
    batch so that dim 0 is ``len(slots)`` contiguous equal groups in the same
    order. ``None`` activates all L slots.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, LoRABankLinear):
            m.set_active_slots(slots)
            n += 1
    if n == 0:
        raise ValueError("set_active_slots: no LoRABankLinear modules found")
    return n


# ---------------------------------------------------------------------------
# Parameters / checkpoints
# ---------------------------------------------------------------------------

def bank_modules(model: nn.Module) -> list[tuple[str, LoRABankLinear]]:
    """``[(dotted_name, module), ...]`` for every bank, in module order."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRABankLinear)]


def bank_parameters(model: nn.Module) -> list[nn.Parameter]:
    """The trainable bank parameters — everything else is frozen by exclusion.

    Under RamTorch nothing can be frozen with ``requires_grad_(False)`` (its
    stages feed every module parameter to ``torch.autograd.grad``), so "frozen"
    means "kept out of the optimizer". The shared norms / modulation tensors
    MUST stay out here: one copy serves every slot, so training them would let
    the adapters contaminate each other.
    """
    params: list[nn.Parameter] = []
    for _, m in bank_modules(model):
        params.append(m.lora_A_bank)
        params.append(m.lora_B_bank)
    return params


def bank_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """The whole bank as one state dict (``*.lora_A_bank`` / ``*.lora_B_bank``)."""
    sd = {}
    for name, m in bank_modules(model):
        sd[f"{name}.lora_A_bank"] = m.lora_A_bank.detach()
        sd[f"{name}.lora_B_bank"] = m.lora_B_bank.detach()
    return sd


def slot_lora_state_dict(model: nn.Module, slot: int) -> dict[str, torch.Tensor]:
    """One slot sliced out under the STANDARD single-LoRA key convention.

    ``*.lora_A`` / ``*.lora_B`` — what `utils/checkpoint.py` and
    `krea2/inference.py --lora-checkpoint` already consume unchanged.
    """
    sd = {}
    for name, m in bank_modules(model):
        if not 0 <= slot < m.n_slots:
            raise IndexError(f"slot {slot} out of range (n_slots={m.n_slots})")
        sd[f"{name}.lora_A"] = m.lora_A_bank[slot].detach()
        sd[f"{name}.lora_B"] = m.lora_B_bank[slot].detach()
    return sd


def bank_budget(model: nn.Module) -> dict:
    """Bank sizing, for the trainer's startup print.

    ``params_per_slot`` is what a single equivalent LoRA would cost, so it is
    the number to compare against `memory/ramtorch_notes.md`'s ~119M figure.
    """
    mods = bank_modules(model)
    if not mods:
        return dict(modules=0, n_slots=0, params_per_slot=0, params_total=0, bytes=0)
    n_slots = mods[0][1].n_slots
    per_slot = sum(
        m.rank * (m.in_features + m.out_features) for _, m in mods
    )
    total = sum(
        m.lora_A_bank.numel() + m.lora_B_bank.numel() for _, m in mods
    )
    nbytes = sum(
        m.lora_A_bank.numel() * m.lora_A_bank.element_size()
        + m.lora_B_bank.numel() * m.lora_B_bank.element_size()
        for _, m in mods
    )
    return dict(
        modules=len(mods),
        n_slots=n_slots,
        params_per_slot=per_slot,
        params_total=total,
        bytes=nbytes,
    )
