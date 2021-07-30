from random import choice
from datasets.utils.tqdm_utils import tqdm
import numpy as np
import torch
from torch import nn
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from itertools import chain
from tqdm import tqdm


tokenizer = GPT2TokenizerFast.from_pretrained("./gpt-opensub")


class RNNModel(torch.nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, rnn_size) -> None:
        super(RNNModel, self).__init__()
        self.seq_size = seq_size
        self.rnn_size = rnn_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.rnn = nn.LSTM(embedding_size, rnn_size, batch_first=False)
        self.dense = nn.Linear(rnn_size, n_vocab)

    def forward(self, x, prev_state):
        # print(x.size())
        embed = self.embedding(x)
        # print(embed.size())
        output, state = self.rnn(embed, prev_state)
        logits = self.dense(output)

        return logits, state

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.rnn_size), torch.zeros(1, batch_size, self.rnn_size))

def predict(device, net: RNNModel, init_tokens, tokenizer, topk=5):
    net.eval()
    h, c = net.zero_state(1)
    h = h.to(device)
    c = c.to(device)
    for w in init_tokens:
        ix = torch.tensor([[w]]).to(device)
        output, (h, c) = net(ix, (h, c))
    _, top_ix = torch.topk(output[0], k=topk)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])
    init_tokens.append(choice)
    for _ in range(100):
        ix = torch.tensor([[choice]]).to(device)
        output, (h, c) = net(ix, (h, c))
        _, top_ix = torch.topk(output[0], k=topk)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        next_token = choice
        if (next_token == tokenizer.encode("<EOS>")[0]):
            break
        init_tokens.append(next_token)
    return tokenizer.decode(init_tokens)
vocab_size = tokenizer.vocab_size + tokenizer.num_special_tokens_to_add()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = RNNModel(vocab_size, 256, 256, 256).to(device)
net.load_state_dict(torch.load("checkpoint_pt/model1-264000.pth"))

with open("test_case.txt") as f:
    test_lines = f.readlines()
    test_lines = [x.replace("\n", "<EOS>") for x in test_lines]
print(test_lines)
with open("result.txt", "w+") as f:
    for line in test_lines:
        tokens = tokenizer.encode(line)
        print(line.replace("<EOS>", ""))
        print(tokens)
        res = predict(device, net, tokens, tokenizer)
        f.write(res + "\n")