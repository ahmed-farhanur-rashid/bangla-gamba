from transformers import PreTrainedTokenizerFast
import json

tokenizer = PreTrainedTokenizerFast.from_pretrained('saved/tokenizer/hf')
vocab = tokenizer.get_vocab()

print("Is '\\n' in vocab directly?", "\\n" in vocab)
print("ID of '\\n':", vocab.get("\n"))
print("Is '<0x0A>' in vocab directly?", "<0x0A>" in vocab)
print("ID of '<0x0A>':", vocab.get("<0x0A>"))
print("Is there any token containing \\n?", [t for t in vocab.keys() if '\n' in t])

# also let's check tokenizer_set/corpus.jsonl
with open('saved/data/tokenizer_set/corpus.jsonl', 'r') as f:
    sample = json.loads(f.readline())
    print("\\n in corpus text?", '\n' in sample.get('text', ''))
