from copy import deepcopy
from typing import Any, List, Tuple

import torch

from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask

from .biasing_layer import ContextBiasingLayer


class CATransformerDecoderLayer(DecoderLayer):
    def __init__(self, biasing_layer_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_biasing_layer = ContextBiasingLayer(**biasing_layer_config)

    def forward(
        self,
        tgt,
        tgt_mask,
        memory,
        memory_mask,
        ct,
        biasing_mask=None,
        cache=None,
    ):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, size).
            memory_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).
            ct (torch.Tensor): Encoded context tokens, float32 (#batch, n_context_tokens, size).
            cache (List[torch.Tensor]): List of cached tensors.
                Each tensor shape should be (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor(#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        residual = tgt
        print('decoder_layer', tgt.shape)
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]

        if biasing_mask is not None:
            biasing_mask = biasing_mask.transpose(-1, -2).repeat(1, 1, ct.size(1))

        x = self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        x = self.label_biasing_layer(
            query=x,
            key=ct,
            value=ct,
            # subsequent_mask is not used in biasing layer
            mask=biasing_mask,
        )
        x = residual + x
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)

        x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask


class CATransformerDecoder(TransformerDecoder):
    """ContextAwareTransformerDecoder
    This class is inherited from espnet Transformer Decoder, so the argument
    is almost the same as that of it.
    """

    def __init__(self, biasing_layer_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        first_block = self.decoders[0]
        self_attn = first_block.self_attn
        size = self_attn.linear_q.weight.shape[0]
        normalize_before = kwargs.get('normalize_before', True)
        self.decoders[0] = CATransformerDecoderLayer(
            biasing_layer_config,
            size=size,
            self_attn=self_attn,
            src_attn=first_block.src_attn,
            feed_forward=first_block.feed_forward,
            dropout_rate=first_block.dropout.p,
            normalize_before=normalize_before,
        )

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        ct: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
            ct (torch.Tensor): Encoded context tokens, float32 (#batch, n_context_tokens, size).

        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        biasing_mask = deepcopy(tgt_mask)

        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(
            memory.device
        )
        # Padding for Longformer
        if memory_mask.shape[-1] != memory.shape[1]:
            padlen = memory.shape[1] - memory_mask.shape[-1]
            memory_mask = torch.nn.functional.pad(
                memory_mask, (0, padlen), "constant", False
            )

        x = self.embed(tgt)
        for layer, decoder in enumerate(self.decoders):
            if layer == 0:
                x, tgt_mask, memory, memory_mask = decoder(
                    x, tgt_mask, memory, memory_mask, ct, biasing_mask,
                )
            else:
                x, tgt_mask, memory, memory_mask = decoder(
                    x, tgt_mask, memory, memory_mask
                )
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)

        olens = tgt_mask.sum(1)
        return x, olens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        ct: torch.Tensor,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            ct (torch.Tensor): Encoded context tokens, float32 (batchm, n_context_tokens, size)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for layer, (c, decoder) in enumerate(zip(cache, self.decoders)):
            if layer == 0:
                x, tgt_mask, memory, memory_mask = decoder(
                    x, tgt_mask, memory, None, ct, cache=c
                )
            else:
                x, tgt_mask, memory, memory_mask = decoder(
                    x, tgt_mask, memory, None, cache=c
                )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache

    def score(self, ys, state, x, ct):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0),
            ys_mask,
            x.unsqueeze(0),
            ct.unsqueeze(0),
            cache=state,
        )
        return logp.squeeze(0), state

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
        ct: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).
            ct (torch.Tensor): Encoded context tokens by a context encoder.

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        print('ys', ys.shape, 'ys_mask', ys_mask.shape, 'xs', xs.shape, 'ct', ct.shape)
        logp, states = self.forward_one_step(ys, ys_mask, xs, ct, cache=batch_state)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list
