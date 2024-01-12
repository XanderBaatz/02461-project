import torch.nn.functional as F
import torch
from torch import Tensor, nn

# Credit: https://github.com/alessandrolamberti/DropBlock
class DropBlock(nn.Module):
    def __init__(self, block_size: int, p: float = 0.5):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.p = p # keep prob, like Dropout


    def calculate_gamma(self, x: Tensor) -> float:
        """Computes gamma, eq. 1 in the paper
        args:
            x (Tensor): Input tensor
        returns:
            float: gamma
        """
        
        to_drop = (1 - self.p) / (self.block_size ** 2)
        to_keep = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return to_drop * to_keep



    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            gamma = self.calculate_gamma(x)
            mask = torch.bernoulli(torch.ones_like(x) * gamma)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum()) # normalize
        return x