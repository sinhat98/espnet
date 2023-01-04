import copy

import torch
from transformers import RobertaForMaskedLM, T5Tokenizer

from espnet2.asr.context_aware.abs_context_encoder import AbsContextEncoder


class RobertaContextEncoder(AbsContextEncoder):
    def __init__(self, checkpoint="rinna/japanese-roberta-base", cache_dir=None):
        super().__init__()
        self.checkpoint = checkpoint
        self.cache_dir = cache_dir
        _model = RobertaForMaskedLM.from_pretrained(
            checkpoint,
            cache_dir=cache_dir,
        )
        for p in _model.parameters():
            p.requires_grad = False
        _model = _model.eval()
        self.token_embedding = copy.deepcopy(_model.roberta.embeddings)
        self.encoder = copy.deepcopy(_model.roberta.encoder)
        del _model

    def forward(self, ct):
        with torch.no_grad():
            context_features = self.encoder(self.token_embedding(ct)).last_hidden_state
        return context_features

    def get_tokenizer(
        self,
    ):
        tokenizer = T5Tokenizer.from_pretrained(
            self.checkpoint,
            cache_dir=self.cache_dir,
        )
        return tokenizer
