from datasets import load_dataset
import torch
from tqdm import tqdm
import torch.nn.functional as F
from rnn_model import RNNModel
from transformers import GPT2TokenizerFast

# dataset = load_dataset("wikipedia", language="vi", date="20210701", split="train")


def batched_perplexity(model:RNNModel, dataset, tokenizer, batch_size, stride):
    device = model.device
    encodings = tokenizer("\n".join(dataset["doc"]), return_tensors="pt")
    text_len = encodings.input_ids.size(1)
    lls = []

    for i in tqdm(range(0, text_len, batch_size * stride)):
        begin_locs, end_locs, trg_lens = [], [], []
        for j in range(batch_size):
            j = i + j * stride
            if j >= text_len:
                break
            begin_loc = max(j + stride - max_len, 0)
            end_loc = min(j + stride, text_len)
            trg_len = end_loc - j  # may be different from stride on last loop

            begin_locs.append(begin_loc)
            end_locs.append(end_loc)
            trg_lens.append(trg_len)

        input_ids = [encodings.input_ids[:, b:e]
                     for b, e in zip(begin_locs, end_locs)]
        target_end_locs = [sen.size(-1) for sen in input_ids]
        input_ids = [
            F.pad(sen, (0, max_len - sen.size(-1)), "constant", 0) for sen in input_ids
        ]  # we dont need attention mask as long as these padded token is not involved in loss calculation
        input_ids = torch.stack(input_ids, dim=1).squeeze(0).to(device)

        # -100 is the default ingore_index value in torch.nn.CrossEntropyLoss
        target_ids = torch.ones_like(input_ids) * -100
        for i, (b, e) in enumerate(zip(trg_lens, target_end_locs)):
            labels = input_ids[i, -b:e].clone()
            target_ids[i, -b:e] = labels

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs["loss"] * sum(trg_lens)

        lls.append(log_likelihood)

    ppl = torch.exp(sum(torch.stack(lls) / end_locs[-1]))
    return ppl


if __name__ == "__main__":
    device = "cpu"
    max_len = 128
    tokenizer = GPT2TokenizerFast.from_pretrained("./gpt-opensub")
    vocab_size = tokenizer.vocab_size + tokenizer.num_special_tokens_to_add()
    net = RNNModel(vocab_size, 256, 256, 256).to(device)
    net.load_state_dict(torch.load("checkpoint_pt/model1-264000.pth"))
    stride = 512
    batch_size = 16
    test = dataset = load_dataset(
        "./open_sub_lm.py", lang="vi", min_len=1, bos="", eos="", split="train[:10%]")
    ppl = batched_perplexity(net, test, tokenizer, batch_size, stride)
    print(f"--------------{ppl=}-------------")
