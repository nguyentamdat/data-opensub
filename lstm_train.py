import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from datasets import load_dataset
import torch.nn.functional as F
import random
import time
import math
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# device = "cpu"
dataset = load_dataset("./open_sub.py", lang="vi", split="train[:1%]")

tokenizer = Tokenizer.from_file("./tokenizer-opensub.json")

BOS_ID, EOS_ID, PAD_ID = tokenizer.encode("<BOS><EOS><PAD>").ids
MAX_LEN = 256


def indexsFromSen(tokenizer: Tokenizer, sen):
    return tokenizer.encode(sen).ids


def tensorFromSen(tokenizer: Tokenizer, sen):
    ids = indexsFromSen(tokenizer, sen)
    ids.append(EOS_ID)
    return torch.tensor(ids, dtype=torch.long, device=device).view(-1, 1)


def tensorFromContext(tokenizer: Tokenizer, context):
    ids = [indexsFromSen(tokenizer, sen) for sen in context]
    out = []
    for id in ids:
        out += id + [EOS_ID]
    return torch.tensor(out, dtype=torch.long, device=device).view(-1, 1)


def tensorFromExample(tokenizer: Tokenizer, item):
    input_tensor = tensorFromContext(tokenizer, item["context"])
    target_tensor = tensorFromSen(tokenizer, item["next_sentence"])
    return (input_tensor, target_tensor)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


teacher_forcing_ratio = 0.5


def train(input_tensor: torch.Tensor, target_tensor: torch.Tensor, encoder: EncoderRNN, decoder: DecoderRNN, encoder_optimizer: torch.optim.Optimizer, decoder_optimizer: torch.optim.Optimizer, criterion, max_len=MAX_LEN):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_len, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[BOS_ID]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_ID:
                break
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


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorFromExample(tokenizer, random.choice(dataset))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        print("start iter", iter)
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


def evaluate(encoder, decoder, inputs, max_length=MAX_LEN):
    with torch.no_grad():
        input_tensor = inputs[0]
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(
            max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[BOS_ID]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        # decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            # decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_ID:
                decoded_words.append(EOS_ID)
                break
            else:
                decoded_words.append(topi.item())

            decoder_input = topi.squeeze().detach()
        decoded_words = tokenizer.decode(decoded_words)
        return decoded_words


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        item = tensorFromExample(tokenizer, random.choice(dataset))
        print('>', tokenizer.decode(item[0].view(1,1,-1)[0][0].tolist()))
        print('=', tokenizer.decode(item[1].view(1,1,-1)[0][0].tolist()))
        output_sentence = evaluate(encoder, decoder, item)
        # output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


if __name__ == "__main__":
    hidden_size = 256
    encoder1 = EncoderRNN(tokenizer.get_vocab_size(), hidden_size).to(device)
    # attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    decoder1 = DecoderRNN(hidden_size, tokenizer.get_vocab_size()).to(device)
    trainIters(encoder1, decoder1, 5, print_every=1)
    evaluateRandomly(encoder1, decoder1, 2)
