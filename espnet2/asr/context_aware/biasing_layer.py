import math

import torch

from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class ContextBiasingLayer(torch.nn.Module):
    """Context biasing layer.

    Args:
        q_dim (int): The dimenssion of a query feature that is assumed as a size of features being biased
        kv_dim (int): The dimenssion of key and value features that is assumed as a size of context features
        out_dim (int): The dimenssion of output.
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        act (str): The type of activation function
                    default: 'gelu'
        dropout_rate (float): Dropout rate.
                    default: 0.0
    """

    def __init__(self, q_dim, kv_dim, out_dim, n_head, act='gelu', dropout_rate=0.0):
        super().__init__()
        assert out_dim % n_head == 0
        self.d_k = out_dim // n_head
        self.h = n_head
        self.linear_q = torch.nn.Linear(q_dim, out_dim)
        self.linear_k = torch.nn.Linear(kv_dim, out_dim)
        self.linear_v = torch.nn.Linear(kv_dim, out_dim)
        self.act = torch.nn.GELU() if act == 'gelu' else get_activation(act)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.q_ln = LayerNorm(q_dim)
        self.attn_ln = LayerNorm(out_dim)
        self.linear_out = torch.nn.Linear(q_dim + out_dim, out_dim)
        self.attn = None

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): audio features or label features
            key (torch.Tensor): context features
            value (torch.Tensor): context features

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, n_context_tokens, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time, d_k).

        """
        n_batch = query.size(0)
        query = self.linear_q(query)
        q = self.act(query).view(n_batch, -1, self.h, self.d_k)
        k = self.act(self.linear_k(key.to(query.dtype))).view(
            n_batch, -1, self.h, self.d_k)
        v = self.act(self.linear_v(value.to(query.dtype))
                     ).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time, d_k)
        k = k.transpose(1, 2)  # (batch, head, n_context_tokens, d_k)
        v = v.transpose(1, 2)  # (batch, head, n_context_tokens, d_k)

        return q, k, v, query

    def forward_attention(self, query, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, n_context_tokens, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time, n_context_tokens).
            mask (torch.Tensor): Mask (#batch, time, n_context_tokens).

        Returns:
            torch.Tensor: Biased features (#batch, time, d_model) which obtained by combining
                            Transformed value (#batch, n_context_tokens, d_model)
                            weighted by the attention score (#batch, time, n_context_tokens)
                                and non-biased features ((#batch, n_context_tokens, d_model))
                            and Transformed.
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, n_context_tokens)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time, n_context_tokens)
        else:
            # (batch, head, time, n_context_tokens)
            self.attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(self.attn)
        attn_output = torch.matmul(p_attn, value)  # (batch, head, time, d_k)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(
                n_batch, -1, self.h * self.d_k)
        )  # (batch, time, d_model)
        attn_output = self.attn_ln(attn_output)
        non_biased_feature = self.q_ln(query)
        biased_feature = torch.concat((attn_output, non_biased_feature), dim=-1)
        return self.linear_out(biased_feature)  # (batch, time, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): audio_features or label_features (#batch, time, size).
            key (torch.Tensor): Key tensor (#batch, n_context_tokens, size).
            value (torch.Tensor): Value tensor (#batch, n_context_tokens, size).
            mask (torch.Tensor): Mask tensor (#batch, n_context_tokens, time).

        Returns:
            torch.Tensor: Output tensor (#batch, time, d_model).

        """
        q, k, v, query = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(query, v, scores, mask)
