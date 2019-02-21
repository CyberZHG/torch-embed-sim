from unittest import TestCase
import torch
import torch.nn as nn
from torch_embed_sim import EmbeddingSim


class TestEmbeddingSim(TestCase):

    def test_sample(self):
        class Net(nn.Module):

            def __init__(self):
                super(Net, self).__init__()
                self.embed = torch.nn.Embedding(num_embeddings=10, embedding_dim=20)
                self.embed_sim = EmbeddingSim(num_embeddings=10, bias=True)

            def forward(self, x):
                return self.embed_sim(self.embed(x), self.embed.weight)

        net = Net()
        print(net)
        x = torch.randint(0, 10, [10, 100]).type(torch.LongTensor)
        y = net(x).argmax(dim=-1)
        batch_size, seq_len = x.size()
        same_count = 0
        for i in range(batch_size):
            for j in range(seq_len):
                if x[i, j] == y[i, j]:
                    same_count += 1
        self.assertGreater(1.0 * same_count / 1000, 0.99)
        EmbeddingSim(num_embeddings=10, bias=False)
