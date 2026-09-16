import math
from typing import Optional


import torch 
from torch import nn 
from distributed_training.model.model_args import DeepSeekV3ModelArgs
from distributed_training.model.moe import FeedForward, MoE
from distributed_training.model.rope import precompute_freqs_cis, apply_rotary_emb
from distributed_training.model.attention import ScaledDotProductionWrapper

class DeepSeekV3Model(nn.Module):
    def __init__(self, args: DeepSeekV3ModelArgs):
        super().__init__()
        self.model_args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(args), persistent=False
        )
        self.layers = nn.ModuleDict({
            str(layer_id): TransformerBlock(layer_id, args) 
            for layer_id in range(args.n_layers)
        })     
        self.norm = nn.RMSNorm(args.dim)
        self.output = nn.Linear(
            args.dim,
            args.vocab_size,
            dtype=torch.get_default_dtype(),
            bias=False,
        )


    def init_weights(
        self,
        init_std: Optional[float],
        buffer_device: Optional[torch.device] = None,
    ):
        """ 
        meta initialization, each gpu owns specific layers, 
        check which layers are owned and perform custom initialization according to model schema
        """
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = precompute_freqs_cis(self.model_args)
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(init_std=init_std, buffer_device=buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5 #big maths
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std, #clamp on weight by 3 std 99.73% of distribution within 3stdvv
                b=cutoff_factor * final_out_std,
            )


        
        