# PyTorch Embedding Similarity

[![Travis](https://travis-ci.org/CyberZHG/torch-embed-sim.svg)](https://travis-ci.org/CyberZHG/torch-embed-sim)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/torch-embed-sim/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/torch-embed-sim)

## Install

```bash
pip install torch-embed-sim
```

## Usage

```python
from torch_embed_sim import EmbeddingSim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.embed = torch.nn.Embedding(num_embeddings=10, embedding_dim=20)
        self.embed_sim = EmbeddingSim(num_embeddings=10)

    def forward(self, x):
        return self.embed_sim(self.embed(x), self.embed.weight)
```
