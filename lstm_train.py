import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import GPT2TokenizerFast
import torch.nn.functional as F
import random
import time
import math
from transformers.file_utils import PaddingStrategy
import numpy as np
from transformers.tokenization_utils_base import TruncationStrategy
import os
from tqdm import tqdm
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# device = "cpu"
dataset = load_dataset("./open_sub.py", lang="vi", split="train")
# MODEL_PATH = "opensub/"
# tokenizer = Tokenizer.from_file(MODEL_PATH + "tokenizer.json")
tokenizer = GPT2TokenizerFast.from_pretrained("./gpt-opensub")

BOS_ID, EOS_ID, PAD_ID, UNK_ID = tokenizer("<BOS><EOS><PAD><PAD>").input_ids
print(BOS_ID, EOS_ID, PAD_ID, UNK_ID)
MAX_LEN = 256


def padding_trunc_left(M, max_len=MAX_LEN):
    maxlen = max(len(r) for r in M)
    axe_y = max_len
    Z = np.zeros((len(M), axe_y), dtype=np.int32)
    for enu, row in enumerate(M):
        if len(row) <= axe_y:
            Z[enu, -len(row):] += row
        else:
            Z[enu, :] += row[-axe_y:]
    return Z


def padding_trunc_right(M, max_len=MAX_LEN):
    maxlen = max_len
    Z = np.zeros((len(M), maxlen), dtype=np.int32)
    for enu, row in enumerate(M):
        Z[enu, :len(row)] += row
    return Z


def indexsFromSen(tokenizer: Tokenizer, sen):
    return padding_trunc_right(tokenizer(sen, return_tensors="np").input_ids)


def tensorFromSen(tokenizer: Tokenizer, sen):
    ids = indexsFromSen(tokenizer, sen)
    return ids


def tensorFromContext(tokenizer: Tokenizer, context):
    return padding_trunc_left(tokenizer(context, return_tensors="np").input_ids)


def tensorFromExample(tokenizer: Tokenizer, item):
    context = ["".join(con) for con in item["context"]]
    item["context"] = context
    input_tensor = tensorFromContext(tokenizer, item["context"])
    target_tensor = tensorFromSen(tokenizer, item["next_sentence"])
    return {"input": input_tensor, "target": target_tensor}


def transform_dataset(tokenizer: Tokenizer):
    def transform(examples):
        return tensorFromExample(tokenizer, examples)
    return transform


dataset.set_transform(transform_dataset(tokenizer))


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, inp, hidden):
        # print(inp.size())
        embedded = self.embedding(inp)
        # print(embedded.size())
        output = embedded.transpose(0, 1)
        # print(output.size())
        # input()
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size=2):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inp, hidden):
        # print(inp.size())
        output = self.embedding(inp).transpose(0, 1)
        # print(output.size())
        # input()
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, batch_size=2):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


teacher_forcing_ratio = 0.5


def train(input_tensor: torch.Tensor, target_tensor: torch.Tensor, encoder: EncoderRNN, decoder: DecoderRNN, encoder_optimizer: torch.optim.Optimizer, decoder_optimizer: torch.optim.Optimizer, criterion, max_len=MAX_LEN, batch_size=2):
    encoder_hidden = encoder.initHidden(batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)

    encoder_outputs = torch.zeros(max_len, encoder.hidden_size, device=device)

    loss = 0
    # print(input_tensor.size(), target_tensor.size(), encoder_hidden.size())

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[:, ei].unsqueeze(1), encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    # print(encoder_outputs.size())

    decoder_input = torch.tensor([[BOS_ID]] * batch_size, device=device)
    decoder_hidden = encoder_hidden

    # print(decoder_input.size())
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # print(target_tensor, target_tensor.size(0))
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            # print(decoder_output.size())
            # print(target_tensor[:, di].size())
            # input()
            loss += criterion(decoder_output, target_tensor[:, di])
            decoder_input = target_tensor[:, di].unsqueeze(1)
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.detach()

            loss += criterion(decoder_output, target_tensor[:, di])
            # print(decoder_input.size())
            # if decoder_input.item() == EOS_ID:
            #     break
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(dataloader, encoder, decoder, n_iters, print_every=1000, learning_rate=0.01, batch_size=2):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)
    train_len = len(dataloader)
    training_pairs = dataloader
    criterion = nn.CrossEntropyLoss(reduction="mean")

    for iter in tqdm(range(1, n_iters + 1), desc="Progress"):
        for training_pair in tqdm(training_pairs):
            input_tensor = training_pair["input"].to(device)
            target_tensor = training_pair["target"].to(device)

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, batch_size=batch_size)
            print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every / train_len
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))


def evaluate(encoder, decoder, inputs, max_length=MAX_LEN, batch_size=1):
    with torch.no_grad():
        input_tensor = inputs["input"].to(device).unsqueeze(0)
        input_length = input_tensor.size(1)
        encoder_hidden = encoder.initHidden(batch_size)

        encoder_outputs = torch.zeros(
            max_length, encoder.hidden_size, device=device)

        # print(input_tensor.size())

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[:, ei].unsqueeze(1),
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[BOS_ID]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        # decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            # print(decoder_input.size())
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            # decoder_attentions[di] = decoder_attention.data
            # print(decoder_output.size())
            topv, topi = decoder_output.data.topk(1)
            # print(topv, topi.item())
            if topi.item() == EOS_ID:
                decoded_words.append(EOS_ID)
                break
            else:
                decoded_words.append(topi.item())

            decoder_input = topi.detach()
        decoded_words = tokenizer.decode(
            decoded_words, skip_special_tokens=True)
        return decoded_words


def evaluateRandomly(dataset, encoder, decoder, n=10):
    for i in range(n):
        item = random.choice(dataset)
        # print(item)
        print('>', tokenizer.decode(
            item["input"].view(1, 1, -1)[0][0].tolist(), skip_special_tokens=True))
        print('=', tokenizer.decode(
            item["target"].view(1, 1, -1)[0][0].tolist(), skip_special_tokens=True))
        output_sentence = evaluate(encoder, decoder, item)
        # output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


if __name__ == "__main__":
    hidden_size = 256
    batch = 128
    # print(dataset[:5])
    vocab_size = tokenizer.vocab_size + tokenizer.num_special_tokens_to_add()
    encoder1 = EncoderRNN(vocab_size, hidden_size).to(device)
    # attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    decoder1 = DecoderRNN(hidden_size, vocab_size).to(device)
    dataset = load_dataset("./open_sub.py", lang="vi", split="train")
    preprocessed = dataset.map(transform_dataset(tokenizer), batched=True)
    preprocessed.set_format(
        type="pt", columns=["input", "target"], device=device)
    dataloader = torch.utils.data.DataLoader(
        preprocessed, batch_size=batch, drop_last=True)
    # print(next(iter(dataloader)))
    print("-------start training----------")
    trainIters(dataloader, encoder1, decoder1,
               10, print_every=1, batch_size=batch)
    print("-------end training------------")
    evaluateRandomly(preprocessed, encoder1, decoder1, 10)
    t = time.localtime()
    strtime = time.strftime("%Y%m%d%H%M%S", t)
    torch.save(encoder1, "encoder-"+strtime + ".pth")
    torch.save(decoder1, "decoder-"+strtime + ".pth")
