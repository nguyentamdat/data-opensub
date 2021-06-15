from tokenizers import Tokenizer
MODEL_PATH = "opensub/"
tok = Tokenizer.from_file(MODEL_PATH + "tokenizer.json")

ids = tok.encode("chào cậu<BOS><PAD><EOS>")
print(ids.ids, ids.tokens)
string = tok.decode(ids.ids)
print(str(string))

