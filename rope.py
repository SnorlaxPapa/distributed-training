import torch
import torch.nn
import math

from distributed_training.model.model_args import DeepSeekV3ModelArgs


def precompute_freqs_cis(args: DeepSeekV3ModelArgs) -> torch.Tensor:   
    """compute frequency cosine im sine for rope embeddings"""
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(base: int, max_seq_length: int, rotations: int, dim: int) -> float:
        """finds dimension given the number of rotations it makes"""
        return (
            dim / (2 * math.log(base)) *
            math.log(max_seq_length / (2 * math.pi * rotations))
        )

    def find_correction_range(
        base: int, max_seq_length: int, 
        dim: int, high_rot: int, low_rot: int
    ) -> tuple[int, int]:
        """given lower and upper bound of rotations, find the boundary of dimensions which lie between rotations"""
        low_dim = math.floor(find_correction_dim(base, max_seq_length, high_rot, dim))
        high_dim = math.ceil(find_correction_dim(base, max_seq_length, low_rot, dim))

        return max(0, low_dim), min(dim - 1, high_dim)

    def linear_ramp(start: float, end: float, dim: int):
        if start == end:
            end += 0.001

        linear_func = (torch.arange(dim, dtype=torch.float32) - start) / (end - start)
        return torch.clamp(linear_func, 0, 1)


    #theta j
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    #yarn for seq len exceeding original seq len
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(base, args.original_seq_len, dim, beta_fast, beta_slow)
        smooth = 1 - linear_ramp(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    #position indiuces
    t = torch.arange(seqlen)

    freqs = torch.outer(t, freqs) #creates rotation frequency for every position, remember the rotation is applied linearly

    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x [b, s, h, d] -> [b, s, h, d/2, 2]
    #freqs_cos = [s, d/2]
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, freqs_cis.size(0), 1, freqs_cis.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)

    return y.to(dtype)