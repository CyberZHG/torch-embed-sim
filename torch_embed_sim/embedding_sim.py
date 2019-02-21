import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['EmbeddingSim']


class EmbeddingSim(nn.Module):

    def __init__(self, num_embeddings, bias=True):
        super(EmbeddingSim, self).__init__()
        self.num_embeddings = num_embeddings
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_embeddings))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x, weight):
        y = x.matmul(weight.transpose(1, 0))
        if self.bias is not None:
            y += self.bias
        return F.softmax(y)

    def extra_repr(self):
        return 'num_embeddings={}, bias={}'.format(
            self.num_embeddings, self.bias is not None,
        )
