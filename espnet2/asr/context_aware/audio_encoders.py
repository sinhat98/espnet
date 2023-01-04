from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

from .biasing_layer import ContextBiasingLayer


class CAConformerEncoder(ConformerEncoder):
    def __init__(self, biasing_layer_config, *args, **kwargs):
        """Initialize
        Args:
            biasing_layer_config (dict): a configuration as follows
                    q_dim (int): a dimenssion of audio features or label fetures
                    kv_dim (int): a dimenssion of context_features
                    out_dim (int): a dimention of output features of this layer
                    n_head (int): the number of attention heads
                    dropout_rate (float): a drop out probability
        """
        super().__init__(*args, **kwargs)
        self.audio_biasing_layer = ContextBiasingLayer(
            **biasing_layer_config,
        )

    def forward(self, context_features, *args, **kwargs):
        audio_features, ilens, _ = super().forward(*args, **kwargs)
        # mask (batch, 1, seq_len)
        biasing_mask = (~make_pad_mask(ilens)[:, None, :]).to(context_features.device)
        # mask (batch, seq_len, n_context_tokens)
        biasing_mask = biasing_mask.transpose(-1, -2).repeat((1, 1, context_features.size(1)))
        audio_biased_features = self.audio_biasing_layer(
            query=audio_features,
            key=context_features,
            value=context_features,
            mask=biasing_mask,
        )  # (Batch, audio_length, n_feat)
        # feature lengths are invariant through biasing layer
        olens = ilens
        return audio_biased_features, olens
