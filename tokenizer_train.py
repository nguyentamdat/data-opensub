from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
import datasets

# batch generator for train


def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["next_sentence"]


# load dataset
dataset = datasets.load_dataset(
    "./open_sub.py", lang="vi", split="train")

print(len(dataset))
print(dataset[-1])

# initialize tokenizer
tokenizer = Tokenizer(models.Unigram())
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoders = decoders.ByteLevel()

# initialize trainer
trainer = trainers.UnigramTrainer(
    vocab_size=52000,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<PAD>", "<BOS>", "<EOS>"],
    show_progress=True
)

# train
tokenizer.train_from_iterator(
    batch_iterator(), trainer=trainer, length=len(dataset))

# save trained tokenizer
tokenizer.save("./tokenizer-opensub.json")
