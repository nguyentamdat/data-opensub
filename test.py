from os import truncate
from transformers import GPT2TokenizerFast
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
import numpy as np
import torch

tokenizer = GPT2TokenizerFast.from_pretrained("./gpt-opensub")
te = [['<EOS>'], ['Khoan, Sóc con.<EOS>']]
inp = ["Đây là câu 1 khá là dài<EOS>", "Và một câu khác khá dài đó ahihi đồ ngốc ngu<EOS>"]

en = tokenizer(inp, return_tensors="np").input_ids
print(en)
def padding(M):
    maxlen = max(len(r) for r in M)
    Z = np.zeros((len(M), maxlen), dtype=np.int32)
    for enu, row in enumerate(M):
        Z[enu, -len(row):] += row
    return Z

en = padding(en)
en = torch.tensor(en)
# en.input_ids = en.input_ids[:,-5:]
# en = tokenizer.tokenize(inp)
print(en, en.size())
print(tokenizer.vocab_size, tokenizer.get_added_vocab(), tokenizer.num_special_tokens_to_add())
# print(tokenizer.batch_decode(en["input_ids"]))
# de = tokenizer.decode(en["input_ids"])
# print(de)