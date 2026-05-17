import textwrap

import torch

from src.config import load_config
from src.prior.quant import Quantizer
from src.quantization import GumbelQuantizer1Bit, GumbelQuantizerTernary
from src.quantization.gumbel_quantizer_1bit import _aln_binary_probabilities
from src.quantization.gumbel_quantizer_ternary import _aln_probabilities


def test_binary_quantizer_backward_replays_forward_noise_on_cpu():
    torch.manual_seed(11)
    q_values = torch.tensor(
        [[0.6, -0.8, 0.3, -0.2], [-0.4, 0.7, -0.5, 0.9]],
        dtype=torch.float32,
    )
    scales = torch.tensor([[0.5, 1.25], [0.75, 0.25]], dtype=torch.float32)
    quantizer = GumbelQuantizer1Bit(
        q_values,
        scales,
        groupsize=2,
        std=0.01,
        strength=2.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        logits_dtype=torch.float32,
    )
    target = torch.tensor(
        [[0.2, -0.3, 0.5, -0.7], [0.4, 0.1, -0.2, 0.6]],
        dtype=torch.float32,
    )

    sign_logits_ref = quantizer.sign_logits.detach().clone().requires_grad_(True)
    scales_ref = quantizer.scales.detach().clone().requires_grad_(True)

    torch.manual_seed(123)
    out = quantizer.forward(temperature=0.7, scale=3.0)
    loss = (out * target).sum()
    loss.backward()

    torch.manual_seed(123)
    u = torch.rand(sign_logits_ref.shape, dtype=torch.float32)
    noise = torch.logit(u, eps=1e-8)
    soft_sign = 2.0 * torch.sigmoid((2.0 * sign_logits_ref * 3.0 + noise) / 0.7) - 1.0
    out_ref = soft_sign * scales_ref[:, quantizer.idx]
    loss_ref = (out_ref * target).sum()
    grad_sign_ref, grad_scales_ref = torch.autograd.grad(loss_ref, (sign_logits_ref, scales_ref))

    torch.testing.assert_close(quantizer.sign_logits.grad, grad_sign_ref)
    torch.testing.assert_close(quantizer.scales.grad, grad_scales_ref)


def test_binary_quantizer_hard_weights_are_pm_scale():
    q_values = torch.tensor([[1.0, -1.0, 1.0, -1.0]], dtype=torch.float32)
    scales = torch.tensor([[0.5, 1.5]], dtype=torch.float32)
    quantizer = GumbelQuantizer1Bit(
        q_values,
        scales,
        groupsize=2,
        std=0.01,
        strength=1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        logits_dtype=torch.float32,
    )

    hard, hard_scales = quantizer.get_hard_weights()

    torch.testing.assert_close(hard.abs(), scales[:, quantizer.idx])
    torch.testing.assert_close(hard_scales, scales)
    assert torch.all(hard != 0)


def test_binary_aln_hard_weights_use_largest_local_score():
    q_values = torch.ones(1, 4, dtype=torch.float32)
    scales = torch.tensor([[0.5, 1.5]], dtype=torch.float32)
    quantizer = GumbelQuantizer1Bit(
        q_values,
        scales,
        groupsize=2,
        std=0.01,
        strength=1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        logits_dtype=torch.float32,
        binary_mode="aln_st",
    )
    with torch.no_grad():
        quantizer.sign_scores.copy_(
            torch.tensor(
                [
                    [[5.0, 1.0, 4.0, 2.0]],
                    [[1.0, 6.0, 3.0, 7.0]],
                ],
                dtype=torch.float32,
            )
        )

    hard, _ = quantizer.get_hard_weights()

    expected = torch.tensor([[-0.5, 0.5, -1.5, 1.5]], dtype=torch.float32)
    torch.testing.assert_close(hard, expected)
    assert quantizer.no_weight_decay_param_names == ("sign_scores",)


def test_binary_aln_softmax_backward_matches_autograd_reference():
    q_values = torch.ones(1, 4, dtype=torch.float32)
    scales = torch.tensor([[1.25]], dtype=torch.float32)
    quantizer = GumbelQuantizer1Bit(
        q_values,
        scales,
        groupsize=4,
        std=0.01,
        strength=1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        logits_dtype=torch.float32,
        binary_mode="aln",
    )
    with torch.no_grad():
        quantizer.sign_scores.copy_(
            torch.tensor(
                [
                    [[1.0, 2.0, 3.0, 4.0]],
                    [[4.0, 3.0, 2.0, 1.0]],
                ],
                dtype=torch.float32,
            )
        )
    target = torch.tensor([[0.2, -0.5, 0.7, -0.3]], dtype=torch.float32)

    scores_ref = quantizer.sign_scores.detach().clone().requires_grad_(True)
    scales_ref = quantizer.scales.detach().clone().requires_grad_(True)

    torch.manual_seed(101)
    out = quantizer.forward(temperature=0.9, scale=50.0)
    loss = (out * target).sum()
    loss.backward()

    torch.manual_seed(101)
    prob = _aln_binary_probabilities(scores_ref, eps=1e-6)
    u = torch.rand(scores_ref.shape, dtype=torch.float32)
    noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    y = torch.softmax((torch.log(prob.clamp_min(1e-6)) + noise) / 0.9, dim=0)
    soft_sign = y[1] - y[0]
    out_ref = soft_sign * scales_ref[:, quantizer.idx]
    loss_ref = (out_ref * target).sum()
    grad_scores_ref, grad_scales_ref = torch.autograd.grad(loss_ref, (scores_ref, scales_ref))

    torch.testing.assert_close(quantizer.sign_scores.grad, grad_scores_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(quantizer.scales.grad, grad_scales_ref, rtol=1e-5, atol=1e-6)


def test_binary_aln_st_backward_matches_autograd_reference():
    q_values = torch.ones(1, 4, dtype=torch.float32)
    scales = torch.tensor([[1.25]], dtype=torch.float32)
    quantizer = GumbelQuantizer1Bit(
        q_values,
        scales,
        groupsize=4,
        std=0.01,
        strength=1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        logits_dtype=torch.float32,
        binary_mode="aln_st",
    )
    with torch.no_grad():
        quantizer.sign_scores.copy_(
            torch.tensor(
                [
                    [[1.0, 2.0, 3.0, 4.0]],
                    [[4.0, 3.0, 2.0, 1.0]],
                ],
                dtype=torch.float32,
            )
        )
    target = torch.tensor([[0.2, -0.5, 0.7, -0.3]], dtype=torch.float32)

    scores_ref = quantizer.sign_scores.detach().clone().requires_grad_(True)
    scales_ref = quantizer.scales.detach().clone().requires_grad_(True)

    torch.manual_seed(202)
    out = quantizer.forward(temperature=0.9, scale=50.0)
    loss = (out * target).sum()
    loss.backward()

    torch.manual_seed(202)
    prob = _aln_binary_probabilities(scores_ref, eps=1e-6)
    u = torch.rand(scores_ref.shape, dtype=torch.float32)
    noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    y_gs = torch.softmax((torch.log(prob.clamp_min(1e-6)) + noise) / 0.9, dim=0)
    y = prob + (y_gs - prob).detach()
    soft_sign = y[1] - y[0]
    out_ref = soft_sign * scales_ref[:, quantizer.idx]
    loss_ref = (out_ref * target).sum()
    grad_scores_ref, grad_scales_ref = torch.autograd.grad(loss_ref, (scores_ref, scales_ref))

    torch.testing.assert_close(quantizer.sign_scores.grad, grad_scores_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(quantizer.scales.grad, grad_scales_ref, rtol=1e-5, atol=1e-6)


def test_symmetric_one_bit_prior_quantizer_has_no_zero_codepoint():
    weights = torch.tensor(
        [[-2.0, -1.0, 0.25, 3.0], [0.5, -0.75, 1.5, -2.5]],
        dtype=torch.float32,
    )
    quantizer = Quantizer()
    quantizer.configure(1, perchannel=True, sym=True, mse=True)
    quantizer.find_params(weights, weight=True)

    quantized = quantizer.quantize(weights)

    assert torch.all(quantized != 0)
    torch.testing.assert_close(quantized.abs(), quantizer.scale.expand_as(weights))


def test_fixed_density_ternary_hard_mask_is_exact_per_row():
    q_values = torch.ones(2, 7, dtype=torch.float32)
    scales = torch.ones(2, 1, dtype=torch.float32)
    quantizer = GumbelQuantizerTernary(
        q_values,
        scales,
        groupsize=7,
        std=0.01,
        strength=1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        logits_dtype=torch.float32,
        mask_mode="fixed_density",
        density=0.5,
        density_scope="row",
    )
    with torch.no_grad():
        quantizer.mask_logits.copy_(
            torch.tensor(
                [[1.0, 7.0, 3.0, 2.0, 5.0, 4.0, 6.0], [7.0, 1.0, 6.0, 2.0, 5.0, 3.0, 4.0]]
            )
        )

    hard, _ = quantizer.get_hard_weights()

    assert (hard != 0).sum(dim=1).tolist() == [3, 3]


def test_fixed_density_ternary_hard_mask_is_exact_per_group():
    q_values = torch.ones(2, 7, dtype=torch.float32)
    scales = torch.ones(2, 3, dtype=torch.float32)
    quantizer = GumbelQuantizerTernary(
        q_values,
        scales,
        groupsize=3,
        std=0.01,
        strength=1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        logits_dtype=torch.float32,
        mask_mode="fixed_density",
        density=0.5,
        density_scope="group",
    )
    with torch.no_grad():
        quantizer.mask_logits.copy_(
            torch.tensor(
                [[1.0, 3.0, 2.0, 9.0, 8.0, 7.0, 6.0], [3.0, 2.0, 1.0, 7.0, 9.0, 8.0, 6.0]]
            )
        )

    hard, _ = quantizer.get_hard_weights()
    nonzero = hard != 0

    assert nonzero[:, 0:3].sum(dim=1).tolist() == [1, 1]
    assert nonzero[:, 3:6].sum(dim=1).tolist() == [1, 1]
    assert nonzero[:, 6:7].sum(dim=1).tolist() == [0, 0]


def test_fixed_density_aln_probabilities_sum_to_budget():
    idx = torch.arange(6) // 3
    scores = torch.ones(2, 6, dtype=torch.float32)

    row_prob = _aln_probabilities(scores, idx, density=0.5, scope="row", eps=1e-6)
    group_prob = _aln_probabilities(scores, idx, density=0.5, scope="group", eps=1e-6)

    torch.testing.assert_close(row_prob.sum(dim=1), torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(group_prob[:, 0:3].sum(dim=1), torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(group_prob[:, 3:6].sum(dim=1), torch.tensor([1.0, 1.0]))


def test_fixed_density_ternary_backward_matches_autograd_when_unclipped():
    q_values = torch.ones(1, 4, dtype=torch.float32)
    scales = torch.tensor([[1.5]], dtype=torch.float32)
    quantizer = GumbelQuantizerTernary(
        q_values,
        scales,
        groupsize=4,
        std=0.01,
        strength=1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        logits_dtype=torch.float32,
        mask_mode="fixed_density",
        density=0.5,
        density_scope="row",
        density_eps=1e-6,
    )
    with torch.no_grad():
        quantizer.mask_logits.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        quantizer.sign_logits.copy_(torch.tensor([[0.3, -0.2, 0.1, -0.4]]))
    target = torch.tensor([[0.2, -0.5, 0.7, -0.3]], dtype=torch.float32)

    mask_scores_ref = quantizer.mask_logits.detach().clone().requires_grad_(True)
    sign_logits_ref = quantizer.sign_logits.detach().clone().requires_grad_(True)
    scales_ref = quantizer.scales.detach().clone().requires_grad_(True)

    torch.manual_seed(321)
    out = quantizer.forward(temperature=0.9, scale=2.5)
    loss = (out * target).sum()
    loss.backward()

    torch.manual_seed(321)
    u_mask = torch.rand(mask_scores_ref.shape, dtype=torch.float32)
    mask_noise = torch.logit(u_mask, eps=1e-8)
    prob = _aln_probabilities(mask_scores_ref, quantizer.idx, density=0.5, scope="row", eps=1e-6)
    soft_mask = torch.sigmoid((torch.logit(prob, eps=1e-6) + mask_noise) / 0.9)

    u_sign = torch.rand(sign_logits_ref.shape, dtype=torch.float32)
    sign_noise = torch.logit(u_sign, eps=1e-8)
    soft_sign = 2.0 * torch.sigmoid((2.0 * sign_logits_ref * 2.5 + sign_noise) / 0.9) - 1.0

    out_ref = soft_mask * soft_sign * scales_ref[:, quantizer.idx]
    loss_ref = (out_ref * target).sum()
    grad_sign_ref, grad_mask_ref, grad_scales_ref = torch.autograd.grad(
        loss_ref,
        (sign_logits_ref, mask_scores_ref, scales_ref),
    )

    torch.testing.assert_close(quantizer.sign_logits.grad, grad_sign_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(quantizer.mask_logits.grad, grad_mask_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(quantizer.scales.grad, grad_scales_ref, rtol=1e-5, atol=1e-6)


def test_config_accepts_fixed_density_ternary_keys(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            quantization:
              gsq_bits: "ternary"
              binary_mode: "aln_st"
              binary_aln_eps: 1.0e-5
              ternary_mask_mode: "fixed_density"
              ternary_density: 0.375
              ternary_density_scope: "group"
              ternary_density_eps: 1.0e-5
            wandb:
              enabled: false
            """
        )
    )

    config = load_config(config_path)

    assert config.quantization.binary_mode == "aln_st"
    assert config.quantization.binary_aln_eps == 1.0e-5
    assert config.quantization.ternary_mask_mode == "fixed_density"
    assert config.quantization.ternary_density == 0.375
    assert config.quantization.ternary_density_scope == "group"
    assert config.quantization.ternary_density_eps == 1.0e-5
