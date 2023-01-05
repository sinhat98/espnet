import numpy as np
from espnet2.train.preprocessor import CommonPreprocessor


class CAPreprocessor(CommonPreprocessor):
    def __init__(
        self, 
        *args,
        context_tokenizer = None,
        context_name = None, 
        **kwargs):
        super().__init__(*args, **kwargs)
        self.context_tokenizer = context_tokenizer
        self.context_name = context_name
    
    def _context_process(self, data):
        if self.context_name in data and self.context_tokenizer is not None:
            context = data[self.context_name]
            context_tokens = self.context_tokenizer.convert_tokens_to_ids(context)
            data[self.context_name] = np.array(context_tokens, dtype=np.int64)

    def __call__(self, uid, data):
        data = self._speech_process(data)
        data = self._text_process(data)
        data = self._context_process(data)
        return data
    
