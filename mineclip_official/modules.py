"""
Adapted from OpenAI CLIP implementation: https://github.com/openai/CLIP
"""
from __future__ import annotations

from collections import OrderedDict
from x_transformers.x_transformers import Encoder,ContinuousTransformerWrapper
import numpy as np
import torch
from torch import nn

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class TemporalTransformer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        embed_dim: int = None,
        depth: int,
        num_heads: int,
        max_seq_len: int,
        # ----- extra tricks, see x_transformers repo ----
        ff_glu=False,
        ff_swish=False,
        attn_one_kv_head=False,
        rel_pos_bias=False,
    ):
        """
        Reference arch:
            bert_base:
                embed_dim = 768
                depth = 12
                num_heads = 12
            bert_large:
                embed_dim = 1024
                depth = 24
                num_heads = 16
        Args:
            input_dim: continuous input feature dimension
            max_seq_len: max sequence length
            embed_dim: embedding dimension, if None, then it is the same as input_dim
                BUT will not add a projection layer from input -> first embedding
                if embed_dim is specified, a projection layer will be added even if
                input_dim == embed_dim
        """
        super().__init__()
        assert isinstance(max_seq_len, int)
        assert isinstance(input_dim, int)
        assert isinstance(depth, int)
        assert isinstance(num_heads, int)

        self.model = ContinuousTransformerWrapper(
            max_seq_len=max_seq_len,
            attn_layers=Encoder(
                dim=input_dim if embed_dim is None else embed_dim,
                depth=depth,
                heads=num_heads,
                ff_glu=ff_glu,
                ff_swish=ff_swish,
                attn_one_kv_head=attn_one_kv_head,
                rel_pos_bias=rel_pos_bias,
            ),
            # if embed_dim is None, do NOT add an input feature projection layer
            dim_in=None if embed_dim is None else input_dim,
            dim_out=None,
        )
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.depth = depth
        self.num_heads = num_heads

    @property
    def output_dim(self):
        return self.input_dim if self.embed_dim is None else self.embed_dim

    def forward(self, x):
        B, L, F = x.size()
        x = self.model(x)
        x = x.mean(dim=1)
        assert x.shape == (B, self.output_dim)
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self._resolution = resolution
        self._patch_size = patch_size
        self._layers = layers
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.cls_token = nn.Parameter(scale * torch.randn(width))
        self.pos_embed = nn.Parameter(
            scale * torch.randn(161, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )
        self.ln_post = nn.LayerNorm(width)
        self.projection = nn.Parameter(scale * torch.randn(width, output_dim))

    # def resize_pos_embed(self, new_resolution):
    #     """
    #     NOTE: call this method AFTER you load pretrained weights!
    #     """
    #     if isinstance(new_resolution, int):
    #         new_resolution = (new_resolution, new_resolution)
    #     else:
    #         assert len(new_resolution) == 2
    #     for r in new_resolution:
    #         assert (
    #             r % self._patch_size == 0
    #         ), f"{new_resolution} is not divisible by {self._patch_size}"

    #     with torch.no_grad():
    #         old_embed = self.pos_embed.data.detach()
    #         cls_embed, old_embed = old_embed[:1], old_embed[1:]
    #         new_embed = interpolate_resize_pos_embed(
    #             old_embed,
    #             self._resolution // self._patch_size,
    #             [r // self._patch_size for r in new_resolution],
    #         )
    #         self.pos_embed = nn.Parameter(torch.cat([cls_embed, new_embed], dim=0))

    def forward(self, x: torch.Tensor):
        bs,ts,c,h,w = x.shape
        x = x.reshape(bs*ts,c,h,w)
        
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B = x.size(0)
        x = x.reshape(B, x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.cls_token.repeat((B, 1, 1)), x], dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.pos_embed
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.blocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.projection is not None:
            x = x @ self.projection

        x = x.reshape(bs,ts,-1)
        return x

class GPT(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        context_length: int,
        vocab_size: int,
        layers: int,
        width: int,
        heads: int,
        is_discrete_text: bool = True,
    ):
        """
        Args:
            is_discrete_text: False to use regular discrete tokens
              True for video sequence of image tokens, and `vocab_size` will be
              interpreted as the dim of each image feature.
        """
        super().__init__()
        self.context_length = context_length
        self._width = width
        self._layers = layers
        self.vocab_size = vocab_size

        self._is_discrete_text = is_discrete_text
        if is_discrete_text:
            self.token_embedding = nn.Embedding(vocab_size, width)
        else:
            self.token_embedding = nn.Linear(vocab_size, width, bias=False)
        self.pos_embed = nn.Parameter(torch.empty(self.context_length, width))
        self.blocks = nn.Sequential(
            *[
                ResidualAttentionBlock(
                    width, heads, attn_mask=self.build_attention_mask()
                )
                for _ in range(layers)
            ]
        )

        self.ln_final = nn.LayerNorm(width)
        self.projection = nn.Parameter(torch.empty(width, embed_dim))

        self.initialize_parameters()

    def initialize_parameters(self):
        if self._is_discrete_text:
            nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.01)

        proj_std = (self._width**-0.5) * ((2 * self._layers) ** -0.5)
        attn_std = self._width**-0.5
        fc_std = (2 * self._width) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.projection is not None:
            nn.init.normal_(self.projection, std=self._width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.pos_embed  # x = x + self.pos_embed[: x.size(1)]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.blocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.projection
        return x

class AdapterHead(nn.Module):
    def __init__(self,video_adapter_layers,text_adapter_layers,feature_dim) -> None:
        super().__init__()
        self.video_adapter_layers = video_adapter_layers
        self.text_adapter_layers = text_adapter_layers
        self.feature_dim = feature_dim

        self.video_residual_weight = None
        self.text_residual_weight = None

        if video_adapter_layers == 0:
            self.video_adapter = nn.Identity()
        else:
            self.video_adapter = nn.Sequential(*([nn.Linear(feature_dim,feature_dim),nn.ReLU()]*(video_adapter_layers-1)),nn.Linear(feature_dim,feature_dim))
            self.video_residual_weight = nn.Parameter(torch.tensor(4.0))

        if text_adapter_layers == 0:
            self.text_adapter = nn.Identity()
        else:
            self.text_adapter = nn.Sequential(*([nn.Linear(feature_dim,feature_dim),nn.ReLU()]*(text_adapter_layers-1)),nn.Linear(feature_dim,feature_dim))
            self.text_residual_weight = nn.Parameter(torch.tensor(4.0))

    def forward(self,video_features,text_features):
        if self.video_residual_weight is None:
            adapted_video = self.video_adapter(video_features)
        else:
            res = torch.sigmoid(self.video_residual_weight)
            adapted_video = res*video_features + (1.0-res)*self.video_adapter(video_features)
        
        if self.text_residual_weight is None:
            adapted_text = self.text_adapter(text_features)
        else:
            res = torch.sigmoid(self.text_residual_weight)
            adapted_text = res*text_features + (1.0-res)*self.text_adapter(text_features)
        return adapted_video,adapted_text


if __name__ == '__main__':
    model =GPT(512,77,49408,12,512,8)
    # model = VisionTransformer(224,16,768,12,12,512)
    for name,para in model.named_parameters():
        print(name,para.shape)