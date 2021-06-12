from random import triangular
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
import datasets

tokenizer = Tokenizer(models.Unigram())
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoders = decoders.ByteLevel()
dataset = datasets.load_dataset("./open_sub.py", lang1="vi", split="train")
trainer = trainers.UnigramTrainer(
    vocab_size=52000,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<PAD>", "<BOS>", "<EOS>"],
)


def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["text"]

tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset))
tokenizer.save("./tokenizer-opensub.json")