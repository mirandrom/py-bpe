# py-bpe
Pure python implementation of byte pair encoding (subword tokenization).

Focuses on:
- Reversible tokenization/encoding;
- Customizable and scalable learning of subword vocabulary from corpus;
- Ease of use in downstream tasks;


## Examples
```python
from py_bpe import BpeTokenizer
from pathlib import Path

savepath = Path("penguin_of_doom.vocab")
corpus = """
    hi every1 im new!!!!!!! *holds up spork* my name is katy but u can call me t3h PeNgU1N oF d00m!!!!!!!! lol…as u can see im very random!!!! thats why i came here, 2 meet random ppl like me ^_^… im 13 years old (im mature 4 my age tho!!) i like 2 watch invader zim w/ my girlfreind (im bi if u dont like it deal w/it) its our favorite tv show!!! bcuz its SOOOO random!!!! shes random 2 of course but i want 2 meet more random ppl =) like they say the more the merrier!!!! lol…neways i hope 2 make alot of freinds here so give me lots of commentses!!!!
    DOOOOOMMMM!!!!!!!!!!!!!!!! <--- me bein random again ^_^ hehe…toodles!!!!!
    love and waffles,
    t3h PeNgU1N oF d00m
"""

learn_bpe_args = dict(
    vocab_size=1000,
    pairable_chars="a-zA-Z0-9",
)

bpet = BpeTokenizer.from_corpus(corpus, savepath, learn_bpe_args=learn_bpe_args)
```
### Load tokenizer
```python
bpet = BpeTokenizer.load(savepath)
```
### Tokenize text
```python
unk_char = "%"
tokens = bpet.tokenize("t3h PeNgU1N oF d00m"+unk_char)
print(tokens)
```
```python
['t3', 'h', ' ', '<maj>', 'pe', '<maj>', 'n', 'g', '<maj>', 'u1', '<maj>', 'n', ' ', 'o', '<maj>', 'f', ' ', 'd0', '0m', '%']
```

### De-tokenize tokens
```python
print(bpet.detokenize(tokens))
```
```python
t3h PeNgU1N oF d00m%
```
### Encode text
```python
indices = bpet.encode("t3h PeNgU1N oF d00m"+unk_char)
print(indices)
```
```python
[91, 24, 12, 5, 76, 5, 21, 34, 5, 96, 5, 21, 12, 15, 5, 28, 12, 98, 100, 0]
```

### Decode indices
```python
print(bpet.decode(indices))
```
```python
t3h PeNgU1N oF d00m<unk>
```
