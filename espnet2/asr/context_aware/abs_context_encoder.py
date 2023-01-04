from abc import ABC, abstractmethod

import torch


class AbsContextEncoder(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, ct: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
