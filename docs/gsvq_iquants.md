# GSVQ for GGUF IQuants

This branch adds the first implementation checkpoint for extending GSQ from
scalar grids to IQuant-style vector quantization.

## Checkpoint 1: Reconstruction Error Decrease

The first required gate is hard reconstruction error decrease while keeping
IQuant scales fixed. Run:

```bash
python scripts/check_gsvq_reconstruction.py --qtype IQ2_XS --vectors 2048 --steps 120 --device cuda
```

For a real Qwen3-4B Unsloth GGUF IQ2 tensor:

```bash
python scripts/check_gsvq_gguf_tensor.py \
  --gguf /home/dalistarh/.cache/huggingface/hub/models--unsloth--Qwen3-4B-GGUF/snapshots/22c9fc8a8c7700b76a1789366280a6a5a1ad1120/Qwen3-4B-UD-Q2_K_XL.gguf \
  --tensor blk.7.ffn_gate.weight \
  --hf-model /home/dalistarh/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --vectors 4096 \
  --steps 100 \
  --device cuda
```

The script exits non-zero if the best hard MSE is not below the initial hard
MSE.

## Implemented

- `src/quantization/gsvq.py`: factorized Gumbel-softmax VQ over magnitude-grid
  choices and sign-pattern choices.
- Shift-style candidate sets: each vector sees candidates from the neighborhood
  of its current code plus nearest candidates to the dense target.
- Fixed scales by construction. The current quantizer learns only discrete
  assignments.
- Optional gradient-only rotation preconditioner inspired by the ICLR 2025
  rotation trick.
- `src/gguf_iq.py`: GGUF tensor discovery, HF name mapping, dense dequant
  loading, and exact IQ2_XXS/IQ2_XS/IQ2_S decomposition into scales and codes.
- `QuantizationTrainer.setup_quantizer_training`: allows a prebuilt GSVQ
  quantizer to plug into the existing trainer without changing scalar GSQ.

## Next Checkpoints

1. Add exact IQ3_XXS/IQ3_S decomposition and repacking. IQ3 uses 4D magnitude
   grid entries interleaved with 8D sign groups, so it needs a dedicated layout.
2. Add acceptance guards to full layer training: write a GSVQ result only when
   hard weighted reconstruction error decreases.
3. Add GGUF byte patching/export. `compressed-tensors` cannot represent IQuant
   formats, so the export path must patch GGUF tensor payloads directly.
4. Run layer-output MSE optimization on Qwen3-4B with 2K+ calibration updates
   after reconstruction-gated tensor optimization passes.

## Qwen3-4B IQuant Hyperparameter Sweep

Model:

- Quantized GGUF: `unsloth/Qwen3-4B-GGUF`, `Qwen3-4B-UD-Q2_K_XL.gguf`
- Dense target: `Qwen/Qwen3-4B`
- Scope: all GGUF IQuant tensors below 4 bits per weight.
- Scales: fixed.
- Acceptance guard: enabled per chunk.

The full sweep results are checked in under:

- `runtime/gsvq_iquant_full`: initial tuned baseline.
- `runtime/gsvq_iquant_hparam`: broad grid over optimizer, LR, steps,
  candidate count, candidate source, temperature/scale schedules, restarts, and
  rotation.
- `runtime/gsvq_iquant_refine`: representative-layer refinement on layers
  0, 7, 14, and 35.
- `runtime/gsvq_iquant_refine_full`: full all-IQuant reruns for the best
  refined schedules.

Best validated full-model setting:

```text
lr=0.12
steps=800
candidate_count=2
neighbor_candidates=1
target_candidates=1
optimizer=adamw
```

Best full all-IQuant reconstruction result:

```text
2.310157277725e-05 -> 2.250513298881e-05
delta = 5.964397884421e-07
relative reduction = 2.5818%
```

Per-layer reductions for the best full run:

```text
layer  0: 1.20755146e-05 -> 1.14087669e-05  rel 5.521%
layer  1: 9.74841706e-06 -> 9.08470515e-06  rel 6.808%
layer  6: 1.36837652e-05 -> 1.30511886e-05  rel 4.623%
layer  7: 3.29003846e-05 -> 3.21911624e-05  rel 2.156%
layer 14: 3.10844024e-05 -> 3.05466178e-05  rel 1.730%
layer 15: 3.03902115e-05 -> 2.98292278e-05  rel 1.846%
layer 16: 1.22848307e-05 -> 1.17325889e-05  rel 4.495%
layer 17: 3.02147840e-05 -> 2.96707647e-05  rel 1.801%
layer 18: 2.96631654e-05 -> 2.91412866e-05  rel 1.759%
layer 35: 1.36336564e-05 -> 1.30155524e-05  rel 4.534%
```

Qtype reductions for the best full run:

```text
IQ2_S   : 3.62147665e-05 -> 3.58239538e-05  rel 1.079%
IQ2_XS  : 4.98964131e-05 -> 4.85112827e-05  rel 2.776%
IQ3_S   : 1.05154873e-05 -> 9.91803014e-06  rel 5.682%
IQ3_XXS : 1.82172836e-05 -> 1.76339504e-05  rel 3.202%
```

Main empirical findings:

- Candidate count mattered more than LR. Two candidates was consistently best.
- One current-neighbor proposal plus one dense-target-distance proposal was the
  best candidate mix. Target-only, neighbor-only, and larger candidate pools
  were worse.
- LR was robust once the two-candidate set was used. `0.12` was slightly best,
  but a wide range worked.
- More steps helped, with diminishing returns. Full all-IQuant reductions were
  `1.1691%`, `1.8195%`, `2.3166%`, `2.4861%`, and `2.5818%` at 50, 100, 200,
  400, and 800 steps respectively.
