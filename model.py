import math
from typing import Optional


import torch 
from torch import nn 
from distributed_training.model.model_args import DeepSeekV3ModelArgs
from distributed_training.model.moe import FeedForward, MoE
from distributed_training.model.rope import precompute_freqs_cis, apply_rotary_emb
from distributed_training.model.attention import ScaledDotProductAttentionWrapper


class Attention(nn.Module):

    def __init__(self, args: DeepSeekV3ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.n_heads = args.n_heads
        self.qk_head_dim = args.qk_rope_head_dim + args.qk_nope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq_a = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=args.norm_eps)
            self.wq_b = nn.Linear(
                self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False
            )
        self.wkv_a = nn.Linear(
            self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=args.norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)
        self.softmax_scale = self.qk_head_dim**-0.5

        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.inner_attention = ScaledDotProductAttentionWrapper()


    @torch.compile
    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        q = self.wq_a(x)
        if self.q_lora_rank > 0:
            q = self.wq_b(self.q_norm(q))

        q = q.view(batch_size, seq_len, -1, self.qk_head_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        #get k
        c = self.wkv_a(x)
        compressed_kv, k_rope = torch.split(
            c, [self.kv_lora_rank, self.qk_rope_head_dim]
        )

        #B, S, rank -> B, S, heads * (k + v) dim
        expanded_kv = self.wkv_b(self.kv_norm(compressed_kv))
        expanded_kv = expanded_kv.view(
            batch_size,
            seq_len,
            -1,
            self.qk_head_dim + self.v_head_dim,
        )

        k_nope, v = torch.split(expanded_kv, [self.qk_head_dim, self.v_head_dim], dim=-1)

        #rope
        q_rope = apply_rotary_emb(q_rope, freq_cis)
        k_rope = apply_rotary_emb(k_rope, freq_cis)


        




class TransformerBlock(nn.Module):

    def __init__(self, layer_id: int, args: DeepSeekV3ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.attention_norm = nn.RMSNorm(args.dim)
        self.ffn_norm = nn.RMSNorm(args.dim)

        self.moe_enabled = layer_id >= args.n_dense_layers
        if self.moe_enabled:
            self.moe = MoE(
                moe_args=args.moe_args,
                hidden_dim=args.dim,
                inter_dim = args.moe_inter_dim,
            )
        else:
            self.feed_forward = FeedForward(
                args.dim, args.inter_dim
            )

        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5 #residual depth scaling
        self.layer_id = layer_id


    def init_weights(self, init_std: Optional[float], buffer_device: Optional[torch.device]) -> None:
        if buffer_device is None:
            raise ValueError(
                'Need buffer device for cis_freq ples'
            )
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(
                init_std=self.weight_init_std, buffer_device=buffer_device
            )
        else:
            self.feed_forward.init_weights(self.weight_init_std)


    @torch.compile
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        x - > norm -> attention -> res -> norm -> ffn -> res
        """

        x = x + self.attention(self.attention_norm(x), freqs_cis)
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))

        return x


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


    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        accept token args as either token ids are embedded depending on pipeline parallelism
        tokens -> hidden layers -> rms norm -> output
        same for output
        """

        x = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        for layers in self.layers.values():
            x = layers(x, self.freqs_cis)
        x = self.norm(x) if self.norm is not None else x
        output = self.output(x) if self.output is not None else x

        return output



        
        