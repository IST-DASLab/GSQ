import os
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from src.gguf_iq import GGUFIQuantStore
from src.quantization.gsvq import (
    FactorizedIQuantGSVQ,
    build_candidate_indices,
    build_synthetic_iquant_problem,
    train_gsvq_reconstruction,
)


LOCAL_QWEN3_GGUF = (
    "/home/dalistarh/.cache/huggingface/hub/models--unsloth--Qwen3-4B-GGUF/"
    "snapshots/22c9fc8a8c7700b76a1789366280a6a5a1ad1120/Qwen3-4B-UD-Q2_K_XL.gguf"
)


class GSVQReconstructionTest(unittest.TestCase):
    def test_synthetic_iq2_xs_reconstruction_decreases(self):
        problem = build_synthetic_iquant_problem(
            qtype_name="IQ2_XS",
            num_vectors=256,
            seed=0,
            device="cpu",
        )
        quantizer = FactorizedIQuantGSVQ(
            problem["target_vectors"],
            problem["scales"],
            problem["magnitude_codebook"],
            problem["sign_codebook"],
            problem["init_magnitude_indices"],
            problem["init_sign_indices"],
            importance=problem["importance"],
            candidate_count=12,
            neighbor_candidates=6,
            target_candidates=6,
        )
        history = train_gsvq_reconstruction(quantizer, steps=60, lr=0.05)
        self.assertLess(history.best_hard_mse, history.initial_hard_mse)

    def test_neighbor_candidates_are_non_self(self):
        codebook = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        target = codebook[[1, 3]]
        init = torch.tensor([1, 3])
        candidates = build_candidate_indices(
            target,
            codebook,
            init,
            candidate_count=2,
            neighbor_candidates=1,
            target_candidates=0,
        )
        self.assertTrue(torch.equal(candidates[:, 0], init))
        self.assertTrue(torch.all(candidates[:, 1] != init))

    def test_posterior_delta_initialization_improves_synthetic_start(self):
        problem = build_synthetic_iquant_problem(
            qtype_name="IQ2_XS",
            num_vectors=256,
            seed=1,
            device="cpu",
        )
        binary = FactorizedIQuantGSVQ(
            problem["target_vectors"],
            problem["scales"],
            problem["magnitude_codebook"],
            problem["sign_codebook"],
            problem["init_magnitude_indices"],
            problem["init_sign_indices"],
            importance=problem["importance"],
            candidate_count=8,
            neighbor_candidates=3,
            target_candidates=4,
        )
        posterior = FactorizedIQuantGSVQ(
            problem["target_vectors"],
            problem["scales"],
            problem["magnitude_codebook"],
            problem["sign_codebook"],
            problem["init_magnitude_indices"],
            problem["init_sign_indices"],
            importance=problem["importance"],
            candidate_count=8,
            neighbor_candidates=3,
            target_candidates=4,
            init_mode="posterior_delta",
            joint_init=True,
        )
        self.assertTrue(torch.isfinite(posterior.magnitude_prior_logits).all())
        self.assertLess(posterior.reconstruction_mse(hard=True), binary.reconstruction_mse(hard=True))

    @unittest.skipUnless(os.path.exists(LOCAL_QWEN3_GGUF), "local Qwen3 GGUF not available")
    def test_local_iq2_xs_decomposition_reconstructs_dense_dequant(self):
        store = GGUFIQuantStore(LOCAL_QWEN3_GGUF)
        decomp = store.decompose("blk.7.attn_k.weight", device="cpu")
        reconstructed = (
            decomp.vector_scales
            * decomp.magnitude_codebook[decomp.magnitude_indices]
            * decomp.sign_codebook[decomp.sign_indices]
        ).reshape(decomp.original_shape)
        self.assertEqual(decomp.qtype_name, "IQ2_XS")
        self.assertTrue(torch.equal(reconstructed, decomp.dense_init))

    @unittest.skipUnless(os.path.exists(LOCAL_QWEN3_GGUF), "local Qwen3 GGUF not available")
    def test_local_iq3_s_decomposition_reconstructs_dense_dequant(self):
        store = GGUFIQuantStore(LOCAL_QWEN3_GGUF)
        decomp = store.decompose("blk.0.ffn_gate.weight", device="cpu")
        reconstructed = (
            decomp.vector_scales
            * torch.cat([
                decomp.half_magnitude_codebook[decomp.first_magnitude_indices],
                decomp.half_magnitude_codebook[decomp.second_magnitude_indices],
            ], dim=-1)
            * decomp.sign_codebook[decomp.sign_indices]
        ).reshape(decomp.original_shape)
        self.assertEqual(decomp.qtype_name, "IQ3_S")
        self.assertTrue(torch.equal(reconstructed, decomp.dense_init))

    @unittest.skipUnless(os.path.exists(LOCAL_QWEN3_GGUF), "local Qwen3 GGUF not available")
    def test_local_iquant_packing_roundtrip_preserves_bytes(self):
        store = GGUFIQuantStore(LOCAL_QWEN3_GGUF)
        names = [
            "blk.7.attn_k.weight",   # IQ2_XS
            "blk.7.ffn_gate.weight", # IQ2_S
            "blk.0.attn_k.weight",   # IQ3_XXS
            "blk.0.ffn_gate.weight", # IQ3_S
        ]
        for name in names:
            with self.subTest(name=name):
                tensor = store.get_tensor(name)
                original = tensor.data.copy()
                writable = SimpleNamespace(data=original.copy())
                decomp = store.decompose(name, device="cpu")
                if decomp.qtype_name == "IQ2_XS":
                    store._pack_iq2_xs(writable, decomp.magnitude_indices, decomp.sign_indices)
                elif decomp.qtype_name == "IQ2_S":
                    store._pack_iq2_s(writable, decomp.magnitude_indices, decomp.sign_indices)
                elif decomp.qtype_name == "IQ3_XXS":
                    store._pack_iq3_xxs(
                        writable,
                        decomp.first_magnitude_indices,
                        decomp.second_magnitude_indices,
                        decomp.sign_indices,
                    )
                elif decomp.qtype_name == "IQ3_S":
                    store._pack_iq3_s(
                        writable,
                        decomp.first_magnitude_indices,
                        decomp.second_magnitude_indices,
                        decomp.sign_indices,
                    )
                else:
                    raise AssertionError(f"Unexpected qtype {decomp.qtype_name}")
                self.assertTrue(np.array_equal(writable.data, original))


if __name__ == "__main__":
    unittest.main()
