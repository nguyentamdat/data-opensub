from torch import nn, tensor
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from typing import List


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

dataset = load_dataset("./open_sub.py", lang="vi", split="train")

tokenizer = GPT2TokenizerFast.from_pretrained("./gpt-opensub")

BOS_ID, EOS_ID, PAD_ID, UNK_ID = tokenizer("<BOS><EOS><PAD><PAD>").input_ids
print(BOS_ID, EOS_ID, PAD_ID, UNK_ID)
MAX_LEN = 256

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, inp, hidden):
        # print(inp.size())
        embedded = self.embedding(inp).unsqueeze(1)
        # print(embedded.size())
        # output = embedded.transpose(0, 1)
        # print(output.size())
        # input()
        output, hidden = self.gru(embedded, hidden)
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
        output = self.embedding(inp)
        # print(output.size())
        # input()
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, batch_size=2):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

def generate(encoder: EncoderRNN, decoder: DecoderRNN, context: List[str], tokenizer: GPT2TokenizerFast):
    context_str = "<EOS>".join(context)
    ids = tokenizer(context_str).input_ids
    if (len(ids) > MAX_LEN):
        ids = ids[-MAX_LEN:]
    # input_len = len(ids)
    hidden = encoder.initHidden(1)
    input_tensor = tensor(ids).to(device)
    encoder_outs, hidden_out = encoder(input_tensor, hidden)
    # print(encoder_outs.size(), hidden_out.size())

    decoder_in = tensor([[BOS_ID]], device=device)
    decoder_ids = []
    for di in range(MAX_LEN):
        decoder_out, hidden_out = decoder(decoder_in, hidden_out)
        topv, topi = decoder_out.data.topk(1)
        if topi.item() in [EOS_ID, PAD_ID]:
            decoder_ids.append(EOS_ID)
            break
        else:
            decoder_ids.append(topi.item())
        decoder_in = topi.detach()
    return decoder_ids

    
encoder = torch.load("encoder-20210628020306.pth")
decoder = torch.load("decoder-20210628020306.pth")

output = generate(encoder, decoder, ["chào bạn", "chào"], tokenizer)
words = tokenizer.decode(output, skip_special_tokens=True)
print(words)

context = []
while True:
    user_input = input("> ")
    context.append(user_input)
    bot_output = generate(encoder, decoder, context, tokenizer)
    words = tokenizer.decode(bot_output, skip_special_tokens=True)
    print("< " + words)
    context.append(words)

