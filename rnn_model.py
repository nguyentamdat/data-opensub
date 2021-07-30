from random import choice
from datasets.utils.tqdm_utils import tqdm
import numpy as np
import torch
from torch import nn
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from itertools import chain
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def get_loss_and_train_op(net: nn.Module, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    return criterion, optimizer


dataset = load_dataset("./open_sub_lm.py", lang="vi",
                       split="train")
map_token = dataset.map(lambda x: tokenizer(x["doc"]), batched=True)
map_token = list(chain.from_iterable(map_token["input_ids"]))


def get_batch(seq_len, batch_size):
    x = map_token
    n_batch = len(x) // (seq_len * batch_size)
    x = x[:n_batch * seq_len * batch_size]
    y = np.zeros_like(x)
    y[:-1] = x[1:]
    y[-1] = x[0]
    x = np.reshape(x, (batch_size, -1))
    y = np.reshape(y, (batch_size, -1))
    # print(x.shape)
    for i in range(0, n_batch * seq_len, seq_len):
        yield x[:, i:i+seq_len], y[:, i:i+seq_len]


batch_size = 16
seq_size = 128
n_batch = len(map_token) // (seq_size * batch_size)
vocab_size = tokenizer.vocab_size + tokenizer.num_special_tokens_to_add()


def main(net, num_step=1, batch_size=1, seq_size=256, grad_norm=5):
    crit, op = get_loss_and_train_op(net, 0.01)

    iteration = 0

    for e in tqdm(range(num_step)):
        batches = get_batch(batch_size, seq_size)
        state_c, state_h = net.zero_state(batch_size)
        state_c = state_c.to(device)
        state_h = state_h.to(device)
        for x, y in tqdm(batches, total=n_batch):
            iteration += 1
            x = torch.tensor(x)
            y = torch.tensor(y)
            x = x.to(device)
            y = y.to(device)

            net.train()

            op.zero_grad()

            logits, (state_h, state_c) = net(x, (state_h, state_c))
            loss = crit(logits.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss_val = loss.item()

            loss.backward()

            _ = nn.utils.clip_grad_norm_(net.parameters(), grad_norm)

            op.step()

            if iteration % 100 == 0:
                print('Epoch: {}/{}'.format(e, num_step),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_val))

            if iteration % 1000 == 0:
                predict(device, net, ["<BOS>"], tokenizer, topk=5)
                torch.save(net.state_dict(),
                        'checkpoint_pt/model1-{}.pth'.format(iteration))


def predict(device, net: RNNModel, init, tokenizer, topk=5):
    net.eval()
    h, c = net.zero_state(1)
    h = h.to(device)
    c = c.to(device)
    for w in init:
        ix = torch.tensor([tokenizer(w).input_ids]).to(device)
        output, (h, c) = net(ix, (h, c))
    _, top_ix = torch.topk(output[0], k=topk)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])
    init.extend(tokenizer.decode([choice]))
    for _ in range(100):
        ix = torch.tensor([[choice]]).to(device)
        output, (h, c) = net(ix, (h, c))
        _, top_ix = torch.topk(output[0], k=topk)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        init.extend(tokenizer.decode([choice]))
    print("".join(init))

net = RNNModel(vocab_size, 256, 256, 256).to(device)
net.load_state_dict(torch.load("checkpoint_pt/model-264000.pth"))
main(net, batch_size=batch_size, seq_size=seq_size, num_step=10)
