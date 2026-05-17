"""Small RNG helpers for replaying stochastic custom-autograd forwards."""

from contextlib import contextmanager

import torch


def get_rng_state(device):
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device=device)
    return torch.get_rng_state()


@contextmanager
def fork_rng_with_state(device, state):
    # The custom autograd functions regenerate the same Gumbel noise in
    # backward, so they need to replay the forward RNG state without advancing
    # the caller's global RNG stream.
    device = torch.device(device)
    if device.type == "cuda":
        with torch.random.fork_rng(devices=[device]):
            torch.cuda.set_rng_state(state, device=device)
            yield
    else:
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(state)
            yield
