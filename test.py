from tokenizers import Tokenizer

tok = Tokenizer.from_file("./tokenizer-opensub.json")


ids = tok.encode("chào cậu").ids
print(ids)
print(tok.decode(ids))
