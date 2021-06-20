from tokenizers.implementations import ByteLevelBPETokenizer
import datasets
import os
from transformers import GPT2TokenizerFast

# batch generator for train
MODEL_PATH = "opensub/"
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)


def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["next_sentence"]


# load dataset
dataset = datasets.load_dataset(
    "./open_sub.py", lang="vi", split="train[:1%]")

print(len(dataset))
print(dataset[-1])

# initialize tokenizer
tokenizer = ByteLevelBPETokenizer()
# tokenizer.normalizer = normalizers.NFC()
# tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
# tokenizer.decoders = decoders.ByteLevel()

# # initialize trainer
# trainer = trainers.BpeTrainer(
#     vocab_size=52000,
#     initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
#     special_tokens=["<PAD>", "<BOS>", "<EOS>"],
#     show_progress=True
# )

# train
tokenizer.train_from_iterator(batch_iterator(), vocab_size=52000, special_tokens=["<PAD>", "<BOS>", "<EOS>"])

# save trained tokenizer
tokenizer.save_model(MODEL_PATH)
tokenizer.save(MODEL_PATH+"tokenizer.json")

ids = tokenizer.encode("chào cậu<BOS><PAD><EOS>")
print(ids.ids, ids.tokens)

fast_token = GPT2TokenizerFast(vocab_file=MODEL_PATH+"vocab.json",merges_file=MODEL_PATH + "merges.txt", tokenizer_file=MODEL_PATH+"tokenizer.json", unk_token="<PAD>", bos_token="<BOS>", eos_token="<EOS>", pad_token="<PAD>")
fast_token.save_pretrained("./gpt-opensub")