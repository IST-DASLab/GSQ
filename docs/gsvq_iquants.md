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
- Exact IQ3_XXS/IQ3_S decomposition and byte repacking for IQ2_XS, IQ2_S,
  IQ2_XXS, IQ3_XXS, and IQ3_S. Repacking preserves fixed GGUF scales and only
  changes discrete assignment payloads.
- Normalized initialization modes for IQuant GSVQ:
  - non-self neighbor candidate generation;
  - local-radius Gaussian shift priors over codebook neighborhoods;
  - row-normalized dense-target likelihood logits;
  - fixed-prior delta modes that do not amplify the prior with the GS logit
    scale schedule;
  - optional joint marginal target initialization across magnitude/sign
    candidates.
- Full-GGUF IQuant patching via `scripts/requantize_gsvq_iquant_gguf.py`.
- Dense-reference KL evaluation for original vs patched GGUF weights via
  `scripts/eval_gguf_kl.py`.
- `QuantizationTrainer.setup_quantizer_training`: allows a prebuilt GSVQ
  quantizer to plug into the existing trainer without changing scalar GSQ.

## Next Checkpoints

1. Add activation-aware or output-KL-aware acceptance. Weight reconstruction
   MSE alone decreased, but end-to-end KL worsened in the full GGUF check below.
2. Add calibration-driven importance weights for the IQuant vector entries.
3. Test partial policies: patch only layers/tensor classes whose local MSE
   decrease also passes an activation-output gate.
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

## Full GGUF Requantization and KL Check

Command:

```bash
python scripts/requantize_gsvq_iquant_gguf.py \
  --gguf /home/dalistarh/.cache/huggingface/hub/models--unsloth--Qwen3-4B-GGUF/snapshots/22c9fc8a8c7700b76a1789366280a6a5a1ad1120/Qwen3-4B-UD-Q2_K_XL.gguf \
  --hf-model /home/dalistarh/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --out-gguf runtime/gsvq_iquant_requantized/Qwen3-4B-UD-Q2_K_XL-gsvq-iq-s800.gguf \
  --out runtime/gsvq_iquant_requantized \
  --steps 800 --lr 0.12 \
  --candidate-count 2 --neighbor-candidates 1 --target-candidates 1 \
  --optimizer-name adamw --weight-decay 0 \
  --temp-start 1.5 --temp-end 0.1 --scale-start 1.0 --scale-end 40.0 \
  --chunk-vectors 65536 --seed 0 --overwrite
```

The output GGUF patches all 45 IQuant tensors below 4 bits in
`Qwen3-4B-UD-Q2_K_XL.gguf`. Each tensor was dequantized after patching and
checked against the accepted reconstruction MSE.

Full patched-GGUF reconstruction result:

```text
2.310157283154e-05 -> 2.250291738853e-05
relative reduction = 2.5914%
```

Per-layer reductions from the patched GGUF:

```text
layer  0: 1.20755147e-05 -> 1.14064041e-05  rel 5.541%
layer  1: 9.74841713e-06 -> 9.08154213e-06  rel 6.841%
layer  6: 1.36837652e-05 -> 1.30486874e-05  rel 4.641%
layer  7: 3.29003847e-05 -> 3.21894437e-05  rel 2.161%
layer 14: 3.10844022e-05 -> 3.05453424e-05  rel 1.734%
layer 15: 3.03902118e-05 -> 2.98267719e-05  rel 1.854%
layer 16: 1.22848307e-05 -> 1.17303501e-05  rel 4.514%
layer 17: 3.02147843e-05 -> 2.96683598e-05  rel 1.808%
layer 18: 2.96631653e-05 -> 2.91392679e-05  rel 1.766%
layer 35: 1.36336565e-05 -> 1.30130590e-05  rel 4.552%
```

End-to-end KL was checked by loading the dense HF Qwen3-4B model as the
reference and loading all 398 GGUF tensors into a second HF model shell. The
candidate model only differs from the original GGUF in the patched IQuant
tensors.

Wikitext2 test, 64 sequences, sequence length 256, 16,320 next-token positions:

```text
original GGUF KL: 0.510352478 nats/token
GSVQ GGUF KL    : 0.519978976 nats/token
delta           : -0.009626498 nats/token (-1.8862% relative)
```

Wikitext2 test, 128 sequences, sequence length 256, 32,640 next-token positions:

```text
original GGUF KL: 0.497392662 nats/token
GSVQ GGUF KL    : 0.508673574 nats/token
delta           : -0.011280912 nats/token (-2.2680% relative)
```

Conclusion: the fixed-scale assignment-only GSVQ pass reliably decreases
IQuant weight reconstruction error, but it did not improve end-to-end KL for
this full Qwen3-4B Unsloth GGUF. On the two Wikitext2 checks above, KL worsened
despite the `2.5914%` IQuant reconstruction MSE reduction.

## Normalized Initialization Ablation

The old initialization used a binary logit bias on the current GGUF code. That
made wider candidate sets hard to use: when candidate count increased from the
effective old `init + target` pair to `init + real neighbors + target`, the
representative-layer reconstruction gain collapsed.

New initialization components:

- candidate neighbors exclude the current code, so `neighbor_candidates=1`
  means one real non-self neighbor;
- local-radius Gaussian prior:
  `-0.5 * ||code_j - code_init||^2 / local_radius(init)`;
- row-normalized target likelihood:
  `-(err_j - min(err)) / mean_j(err_j - min(err))`;
- `*_delta` modes keep the prior/likelihood logits fixed and train only a
  delta, avoiding late amplification of the prior by the GS logit-scale
  schedule;
- joint marginal initialization scores magnitude/sign candidates by minimizing
  over the other factor before initializing factorized logits.

Representative ablation: layers `0,7,14,35`, first `8192` vectors per tensor,
100 steps, `lr=0.12`.

```text
target_delta_joint_c3_n1_t1       rel 3.702%
target_delta_c3_n1_t1             rel 3.566%
posterior_delta_c3_n1_t1_pw025    rel 3.538%
binary_c2_t1                      rel 3.373%
posterior_delta_c3_n1_t1          rel 3.349%
posterior_c3_n1_t1                rel 3.272%
target_delta_c4_n1_t2             rel 3.000%
target_delta_c5_n2_t2             rel 2.861%
posterior_c8_n4_t4                rel 2.641%
posterior_delta_c8_n4_t4          rel 2.599%
binary_c8_n4_t4                   rel 0.208%
```

Full all-IQuant run for the best representative initialization:

```text
init_mode=target_delta
joint_init=true
candidate_count=3
neighbor_candidates=1
target_candidates=1
steps=800
lr=0.12

2.310157283154e-05 -> 2.247843325782e-05
relative reduction = 2.6974%
```

This beats the previous best full reconstruction result (`2.5818%` in the
read-only sweep, `2.5914%` in the patched-GGUF run). The improvement came
mostly from IQ2_XS attention tensors; wide binary neighbor search remained bad.

End-to-end KL still worsened after patching the full GGUF:

```text
Wikitext2, 64 sequences, 16,320 positions:
original GGUF KL: 0.510352478 nats/token
new GSVQ GGUF KL: 0.527014296 nats/token
delta           : -0.016661818 nats/token (-3.2648% relative)

Wikitext2, 128 sequences, 32,640 positions:
original GGUF KL: 0.497392662 nats/token
new GSVQ GGUF KL: 0.518081333 nats/token
delta           : -0.020688671 nats/token (-4.1594% relative)
```

Conclusion: normalized initialization fixes the immediate neighbor-search
problem and improves full fixed-scale reconstruction, but it makes the
end-to-end KL problem more obvious. The next gate should be activation/KL-aware
acceptance before patching a tensor, not more weight-MSE-only optimization.
